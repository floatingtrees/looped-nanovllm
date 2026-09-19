import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.block_size = config.kvcache_block_size
        self.rows: list[Sequence] = []
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def entry(self, seq: Sequence) -> tuple:
        kv_slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        return (seq.last_token, len(seq) - 1, kv_slot, len(seq), seq.block_table, seq.temperature)

    def remove_rows(self, seqs: list[Sequence]):
        removed = {seq.row for seq in seqs if seq.row is not None}
        for seq in seqs:
            seq.row = None
        if not removed:
            return
        n = len(self.rows)
        new_n = n - len(removed)
        holes = sorted(row for row in removed if row < new_n)
        movers = [row for row in range(new_n, n) if row not in removed]
        moves = list(zip(movers, holes))
        for src, dst in moves:
            self.rows[dst] = self.rows[src]
            self.rows[dst].row = dst
        del self.rows[new_n:]
        self.model_runner.call("remove", moves, new_n)

    def release(self) -> list[Sequence]:
        finished, preempted = self.scheduler.take_released()
        self.remove_rows(finished + preempted)
        return finished

    def step(self):
        seqs = self.scheduler.schedule_prefill()
        if seqs:
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            token_ids, logprobs = self.model_runner.call("prefill", seqs)
            continuing = self.scheduler.allocate_slots(self.scheduler.postprocess_prefill(seqs, token_ids, logprobs))
            finished = self.release()
            if continuing:
                start = self.model_runner.call("admit", [self.entry(seq) for seq in continuing])
                for i, seq in enumerate(continuing):
                    seq.row = start + i
                self.rows.extend(continuing)
        else:
            assert self.rows
            self.model_runner.call("run_pass")
            exited, depths = self.model_runner.call("exit_rows_after_pass")
            num_tokens = -len(exited)
            if exited:
                token_ids, logprobs = self.model_runner.call("finish", exited)
                seqs = [self.rows[row] for row in exited]
                continuing = self.scheduler.allocate_slots(self.scheduler.commit(seqs, token_ids, depths, logprobs))
                if continuing:
                    self.model_runner.call("restart", [seq.row for seq in continuing], [self.entry(seq) for seq in continuing])
            finished = self.release()
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.completion_depths, seq.completion_logprobs) for seq in finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            elif num_tokens < 0:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, depths, logprobs in output:
                outputs[seq_id] = (token_ids, depths, logprobs)
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids, "depths": depths, "logprobs": logprobs}
                   for token_ids, depths, logprobs in outputs]
        return outputs
