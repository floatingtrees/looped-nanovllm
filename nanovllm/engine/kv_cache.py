import torch
import triton
import triton.language as tl


@triton.jit
def fill_forward_kernel(
    kv_ptr,
    rows_ptr,
    count_ptr,
    kv_slot_ptr,
    depth_ptr,
    plane_stride,
    block_size,
    num_depths,
    width,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0)
    if lane >= tl.load(count_ptr):
        return
    row = tl.load(rows_ptr + lane).to(tl.int64)
    depth = tl.load(depth_ptr + row).to(tl.int64)
    target = depth + 1 + tl.program_id(2)
    if target >= num_depths:
        return
    slot = tl.load(kv_slot_ptr + row).to(tl.int64)
    block = slot // block_size
    offset = slot % block_size
    plane = tl.program_id(1).to(tl.int64) * plane_stride
    src = plane + ((block * num_depths + depth) * block_size + offset) * width
    dst = plane + ((block * num_depths + target) * block_size + offset) * width
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    tl.store(kv_ptr + dst + cols, tl.load(kv_ptr + src + cols, mask=mask), mask=mask)


class KVCache:
    """The whole kv cache, allocated once at startup.

    Indexed by (layer, depth): the layer is chosen in code, one slice per attention
    layer, while the loop depth is data, carried per row. Each logical block the
    BlockManager hands out owns num_depths physical blocks, one per depth:

        physical block = logical block * num_depths + depth

    so a batch can hold rows at different depths, and a model that is not looped
    (num_depths == 1) addresses the cache exactly as before.
    """

    def __init__(
        self,
        num_layers: int,
        num_depths: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        self.num_depths = num_depths
        self.block_size = block_size
        self.kv = torch.empty(2, num_layers, num_blocks * num_depths, block_size, num_kv_heads, head_dim)

    def layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv[0, layer_id], self.kv[1, layer_id]

    def slot_mapping(self, slot_mapping: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Logical slots -> physical slots at each row's depth. -1 (skip) stays -1."""
        if self.num_depths == 1:
            return slot_mapping
        block, offset = slot_mapping // self.block_size, slot_mapping % self.block_size
        physical = (block * self.num_depths + depth) * self.block_size + offset
        return torch.where(slot_mapping < 0, slot_mapping, physical).to(slot_mapping.dtype)

    def block_tables(self, block_tables: torch.Tensor | None, depth: torch.Tensor) -> torch.Tensor | None:
        """Logical block tables -> physical ones at each sequence's depth. Padding stays -1."""
        if block_tables is None or self.num_depths == 1:
            return block_tables
        physical = block_tables * self.num_depths + depth.unsqueeze(1)
        return torch.where(block_tables < 0, block_tables, physical).to(block_tables.dtype)

    def fill_forward(
        self,
        rows: torch.Tensor,
        count: torch.Tensor,
        kv_slots: torch.Tensor,
        depth: torch.Tensor,
    ):
        width = self.kv.size(-2) * self.kv.size(-1)
        planes = self.kv.size(0) * self.kv.size(1)
        grid = (rows.size(0), planes, self.num_depths - 1)
        fill_forward_kernel[grid](
            self.kv, rows, count, kv_slots, depth,
            self.kv.stride(1), self.block_size, self.num_depths, width,
            BLOCK=triton.next_power_of_2(width),
        )
