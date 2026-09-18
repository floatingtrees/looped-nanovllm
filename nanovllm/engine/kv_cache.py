import torch


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
