import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.kv_cache import KVCache
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
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
        self.kv_cache = None    # allocated after warmup has measured peak activation memory
        self.depth_buffer = torch.zeros(max(config.max_num_batched_tokens, config.max_num_seqs), dtype=torch.int64)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
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
            del self.graphs, self.graph_pool
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

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        depth = self.run_prelude(seqs, True)
        for _ in range(self.num_loops):
            gate_logits = self.run_recurrence(depth)
            depth += 1
        self.run_coda()
        self.has_gate = gate_logits is not None
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

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, kv_cache=self.kv_cache)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_prelude(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        self.temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        self.positions = positions
        self.num_rows = num_rows = input_ids.size(0)
        self.graph = None
        if not (is_prefill or self.enforce_eager or num_rows > 512):
            context = get_context()
            self.graph = self.graphs[next(x for x in self.graph_bs if x >= num_rows)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:num_rows] = input_ids
            graph_vars["positions"][:num_rows] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:num_rows] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:num_rows] = context.context_lens
            graph_vars["block_tables"][:num_rows, :context.block_tables.size(1)] = context.block_tables
            self.graph["prelude"].replay()
            self.hidden_states = graph_vars["hidden_states"][:num_rows]
        else:
            self.hidden_states = self.model.prelude(input_ids)
        return self.depth_buffer[:num_rows].zero_()

    @torch.inference_mode()
    def run_recurrence(self, depth: torch.Tensor) -> torch.Tensor | None:
        if self.graph is not None:
            captured = self.graph_vars["depth"][:self.num_rows]
            if depth.data_ptr() != captured.data_ptr():
                captured.copy_(depth)
            self.graph["recurrence"].replay()
            return self.graph_vars["gate_logits"][:self.num_rows] if self.has_gate else None
        self.hidden_states, gate_logits = self.model.recurrence(self.hidden_states, self.positions, depth)
        return gate_logits.squeeze(-1) if gate_logits is not None else None

    @torch.inference_mode()
    def run_coda(self) -> list[int] | None:
        if self.graph is not None and self.graph["coda"] is not None:
            self.graph["coda"].replay()
            logits = self.graph_vars["logits"][:self.num_rows]
        else:
            logits = self.model.coda(self.hidden_states)
        token_ids = self.sampler(logits, self.temperatures).tolist() if self.rank == 0 else None
        self.hidden_states = self.positions = self.temperatures = self.graph = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        depth = self.depth_buffer[:max_bs]
        hidden_states = torch.zeros(max_bs, hf_config.hidden_size)
        gate_logits = torch.zeros(max_bs)
        logits = torch.zeros(max_bs, hf_config.vocab_size) if self.world_size == 1 else None
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        def capture(fn):
            graph = torch.cuda.CUDAGraph()
            fn()
            with torch.cuda.graph(graph, self.graph_pool):
                fn()
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            return graph

        for bs in reversed(self.graph_bs):
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], kv_cache=self.kv_cache)

            def prelude():
                hidden_states[:bs] = self.model.prelude(input_ids[:bs])

            def recurrence():
                output, gate = self.model.recurrence(hidden_states[:bs], positions[:bs], depth[:bs])
                hidden_states[:bs] = output
                if gate is not None:
                    gate_logits[:bs] = gate.squeeze(-1)

            def coda():
                logits[:bs] = self.model.coda(hidden_states[:bs])

            self.graphs[bs] = {
                "prelude": capture(prelude),
                "recurrence": capture(recurrence),
                "coda": capture(coda) if logits is not None else None,
            }
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            depth=depth,
            hidden_states=hidden_states,
            gate_logits=gate_logits,
            logits=logits,
        )
