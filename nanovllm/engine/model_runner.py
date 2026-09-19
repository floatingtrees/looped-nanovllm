import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.kv_cache import KVCache
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
from nanovllm.layers.row_ops import copy_rows, fill_rows
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.ouro import OuroForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


MODEL_REGISTRY = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "OuroForCausalLM": OuroForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = MODEL_REGISTRY[hf_config.architectures[0]](hf_config)
        load_model(self.model, config.model)
        self.num_loops = getattr(hf_config, "total_ut_steps", 1)
        self.exit_policy = config.exit_policy
        self.exit_threshold = config.exit_threshold
        self.max_batch_size = config.max_batch_size
        self.graph_bs = sorted({b for b in [1, 2, 4, 8] + list(range(16, self.max_batch_size + 1, 16)) if b <= self.max_batch_size} | {self.max_batch_size})
        self.kv_cache = None    # allocated after warmup has measured peak activation memory
        self.allocate_pool()
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self.build_launchers()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool, self.pass_launch, self.finish_launch, self.write_launch, self.remove_launch
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def allocate_pool(self):
        hf_config = self.config.hf_config
        size = self.max_batch_size
        self.max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        self.n = 0
        self.row_depth = []
        self.hidden_states = torch.zeros(size, hf_config.hidden_size)
        self.positions = torch.zeros(size, dtype=torch.int64)
        self.kv_slots = torch.full((size,), -1, dtype=torch.int32)
        self.context_lens = torch.zeros(size, dtype=torch.int32)
        self.block_tables = torch.zeros(size, self.max_blocks, dtype=torch.int32)
        self.depth = torch.zeros(size, dtype=torch.int64)
        self.remaining = torch.ones(size, dtype=torch.float32)
        self.temperatures = torch.ones(size, dtype=torch.float32)
        self.exit_mask = torch.zeros(size, dtype=torch.bool)
        self.exit_rows = torch.zeros(size, dtype=torch.int64)
        self.exit_count = torch.zeros(1, dtype=torch.int32)
        self.exit_tokens = torch.zeros(size, dtype=torch.int64)
        self.exit_logprobs = torch.zeros(size, dtype=torch.float32)
        self.coda_input = torch.zeros(size, hf_config.hidden_size)
        self.write_int = torch.zeros(size, 5 + self.max_blocks, dtype=torch.int64)
        self.write_float = torch.zeros(size, dtype=torch.float32)
        self.write_count = torch.zeros(1, dtype=torch.int32)
        self.move_int = torch.zeros(size, 2, dtype=torch.int64)
        self.move_count = torch.zeros(1, dtype=torch.int32)
        self.vacate_rows = torch.zeros(size, dtype=torch.int64)
        self.vacate_count = torch.zeros(1, dtype=torch.int32)
        self.lanes = torch.arange(size, dtype=torch.int64)
        self.prefill_depth = torch.zeros(self.config.max_num_batched_tokens, dtype=torch.int64)
        self.staging = {
            name: torch.zeros(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
            for name, tensor in (
                ("exit_rows", self.exit_rows), ("exit_count", self.exit_count),
                ("write_int", self.write_int), ("write_float", self.write_float), ("write_count", self.write_count),
                ("move_int", self.move_int), ("move_count", self.move_count),
                ("vacate_rows", self.vacate_rows), ("vacate_count", self.vacate_count),
            )
        }
        self.staging_numpy = {name: tensor.numpy() for name, tensor in self.staging.items()}

    def upload(self, name: str, rows: int):
        getattr(self, name)[:rows].copy_(self.staging[name][:rows], non_blocking=True)

    def bucket(self, n: int) -> int:
        return next(b for b in self.graph_bs if b >= n)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.prefill(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        attention_layers = [module for module in self.model.modules() if isinstance(module, Attention)]
        for layer_id, module in enumerate(attention_layers):
            module.layer_id = layer_id
        num_depths = self.num_loops
        block_bytes = 2 * len(attention_layers) * num_depths * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = KVCache(len(attention_layers), num_depths, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, self.kv_cache)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def prefill(self, seqs: list[Sequence]) -> tuple[list[int], list[float]]:
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs)
        depth = self.prefill_depth[:input_ids.size(0)]
        hidden_states = self.model.prelude(input_ids)
        for ut_step in range(self.num_loops):
            depth.fill_(ut_step)
            hidden_states, gate_logits = self.model.recurrence(hidden_states, positions, depth)
        self.has_gate = gate_logits is not None
        logits = self.model.coda(hidden_states)
        token_ids = self.sampler(logits, temperatures)
        logprobs = self.logprobs(logits, token_ids)
        reset_context()
        return token_ids.tolist(), logprobs.tolist()

    def logprobs(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        return logits.gather(1, token_ids.unsqueeze(1)).squeeze(1) - torch.logsumexp(logits, dim=-1)

    def stage_rows(self, rows: list[int], entries: list[tuple]):
        k = len(rows)
        write_int, write_float = self.staging_numpy["write_int"], self.staging_numpy["write_float"]
        for i, (row, (token_id, position, kv_slot, context_len, block_table, temperature)) in enumerate(zip(rows, entries)):
            write_int[i, :5] = (row, token_id, position, kv_slot, context_len)
            write_int[i, 5:5 + len(block_table)] = block_table
            write_float[i] = temperature
        self.staging_numpy["write_count"][0] = k
        self.upload("write_int", k)
        self.upload("write_float", k)
        self.upload("write_count", 1)

    @torch.inference_mode()
    def admit(self, entries: list[tuple]) -> int:
        start = self.n
        assert start + len(entries) <= self.max_batch_size
        rows = list(range(start, start + len(entries)))
        self.stage_rows(rows, entries)
        self.write_launch()
        self.n += len(entries)
        self.row_depth.extend([0] * len(entries))
        return start

    @torch.inference_mode()
    def restart(self, rows: list[int], entries: list[tuple]):
        self.stage_rows(rows, entries)
        self.write_launch()
        for row in rows:
            self.row_depth[row] = 0

    @torch.inference_mode()
    def run_pass(self):
        self.pass_launch[self.bucket(self.n)]()

    @torch.inference_mode()
    def exit_rows_after_pass(self) -> tuple[list[int], list[int]]:
        n = self.n
        if self.exit_policy != "never" and self.has_gate:
            exits = self.exit_mask[:n].tolist()
        else:
            exits = [depth == self.num_loops - 1 for depth in self.row_depth]
        rows, depths = [], []
        for row, exited in enumerate(exits):
            if exited:
                rows.append(row)
                depths.append(self.row_depth[row])
            else:
                self.row_depth[row] += 1
        return rows, depths

    @torch.inference_mode()
    def finish(self, rows: list[int]) -> tuple[list[int], list[float]]:
        k = len(rows)
        self.staging_numpy["exit_rows"][:k] = rows
        self.staging_numpy["exit_count"][0] = k
        self.upload("exit_rows", k)
        self.upload("exit_count", 1)
        self.finish_launch[self.bucket(k)]()
        return self.exit_tokens[:k].tolist(), self.exit_logprobs[:k].tolist()

    @torch.inference_mode()
    def remove(self, moves: list[tuple[int, int]], new_n: int):
        m, old_n = len(moves), self.n
        vacate = old_n - new_n
        if moves:
            self.staging_numpy["move_int"][:m] = moves
        self.staging_numpy["move_count"][0] = m
        self.staging_numpy["vacate_rows"][:vacate] = range(new_n, old_n)
        self.staging_numpy["vacate_count"][0] = vacate
        self.upload("move_int", m)
        self.upload("move_count", 1)
        self.upload("vacate_rows", vacate)
        self.upload("vacate_count", 1)
        self.remove_launch()
        for src, dst in moves:
            self.row_depth[dst] = self.row_depth[src]
        del self.row_depth[new_n:]
        self.n = new_n

    def pass_fn(self, b: int):
        def run():
            hidden_states, gate_logits = self.model.recurrence(self.hidden_states[:b], self.positions[:b], self.depth[:b])
            self.hidden_states[:b] = hidden_states
            depth = self.depth[:b]
            last = depth == self.num_loops - 1
            if self.exit_policy == "threshold" and gate_logits is not None:
                remaining = self.remaining[:b]
                remaining.mul_(torch.where(last, 1.0, 1.0 - torch.sigmoid(gate_logits.squeeze(-1).float())))
                exits = last | (1.0 - remaining >= self.exit_threshold)
            elif self.exit_policy == "sample" and gate_logits is not None:
                hazard = torch.sigmoid(gate_logits.squeeze(-1).float())
                exits = last | (torch.rand_like(hazard) < hazard)
            else:
                exits = last
            self.exit_mask[:b] = exits
            depth.add_((~exits).to(depth.dtype))
        context = dict(slot_mapping=self.kv_slots[:b], context_lens=self.context_lens[:b], block_tables=self.block_tables[:b])
        return run, context

    def finish_fn(self, b: int):
        def run():
            rows = self.exit_rows[:b]
            torch.index_select(self.hidden_states, 0, rows, out=self.coda_input[:b])
            logits = self.model.coda(self.coda_input[:b])
            token_ids = self.sampler(logits, self.temperatures.index_select(0, rows))
            self.exit_tokens[:b] = token_ids
            self.exit_logprobs[:b] = self.logprobs(logits, token_ids)
            if self.exit_policy != "never":
                self.model.early_exit_protocol(rows, self.exit_count, self.kv_slots, self.depth)
        return run, {}

    def write_fn(self):
        size = self.max_batch_size
        rows, count, lanes = self.write_int[:, 0], self.write_count, self.lanes

        def run():
            copy_rows(self.positions, self.write_int[:, 2:3], rows, lanes, count, size)
            copy_rows(self.kv_slots, self.write_int[:, 3:4], rows, lanes, count, size)
            copy_rows(self.context_lens, self.write_int[:, 4:5], rows, lanes, count, size)
            copy_rows(self.block_tables, self.write_int[:, 5:], rows, lanes, count, size)
            copy_rows(self.temperatures, self.write_float, rows, lanes, count, size)
            fill_rows(self.depth, rows, count, 0, size)
            fill_rows(self.remaining, rows, count, 1, size)
            self.model.prelude_into(self.hidden_states, rows, self.write_int[:, 1], count)
        return run, {}

    def remove_fn(self):
        size = self.max_batch_size
        src, dst, count = self.move_int[:, 0], self.move_int[:, 1], self.move_count
        buffers = (self.hidden_states, self.positions, self.kv_slots, self.context_lens,
                   self.block_tables, self.depth, self.remaining, self.temperatures)

        def run():
            for buffer in buffers:
                copy_rows(buffer, buffer, dst, src, count, size)
            fill_rows(self.kv_slots, self.vacate_rows, self.vacate_count, -1, size)
            fill_rows(self.context_lens, self.vacate_rows, self.vacate_count, 0, size)
        return run, {}

    def launcher(self, fn, context: dict):
        def eager():
            set_context(False, kv_cache=self.kv_cache, **context)
            fn()
            reset_context()
        if self.enforce_eager:
            return eager
        set_context(False, kv_cache=self.kv_cache, **context)
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.graph_pool):
            fn()
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        reset_context()
        self.graphs.append(graph)
        return graph.replay

    @torch.inference_mode()
    def build_launchers(self):
        self.graphs = []
        self.graph_pool = None
        self.pass_launch, self.finish_launch = {}, {}
        for b in reversed(self.graph_bs):
            self.pass_launch[b] = self.launcher(*self.pass_fn(b))
            self.finish_launch[b] = self.launcher(*self.finish_fn(b))
        self.write_launch = self.launcher(*self.write_fn())
        self.remove_launch = self.launcher(*self.remove_fn())
        self.depth.zero_()
        self.remaining.fill_(1)
        self.exit_mask.zero_()
