from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    kv_cache: "KVCache | None" = None
    # physical kv cache addresses at the loop depth being run, see set_depth
    kv_slot_mapping: torch.Tensor | None = None
    kv_block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, kv_cache=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, kv_cache)
    if kv_cache is not None and kv_cache.num_depths == 1:
        # with a single depth, logical and physical addresses coincide
        _CONTEXT.kv_slot_mapping, _CONTEXT.kv_block_tables = slot_mapping, block_tables

def set_depth(depth: torch.Tensor):
    """Point attention at the kv cache for `depth`, one loop depth per row."""
    context = _CONTEXT
    if context.kv_cache is None:
        return
    context.kv_slot_mapping = context.kv_cache.slot_mapping(context.slot_mapping, depth)
    if context.block_tables is not None:
        # block tables are per sequence; in prefill a sequence's rows share its depth
        seq_depth = depth[context.cu_seqlens_q[:-1]] if context.is_prefill else depth
        context.kv_block_tables = context.kv_cache.block_tables(context.block_tables, seq_depth)
    else:
        context.kv_block_tables = None

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
