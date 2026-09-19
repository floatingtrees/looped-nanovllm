import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_kernel(
    dst_ptr,
    dst_stride,
    src_ptr,
    src_stride,
    dst_index_ptr,
    dst_index_stride,
    src_index_ptr,
    src_index_stride,
    count_ptr,
    width,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0)
    if lane >= tl.load(count_ptr):
        return
    dst_row = tl.load(dst_index_ptr + lane * dst_index_stride).to(tl.int64)
    src_row = tl.load(src_index_ptr + lane * src_index_stride).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    values = tl.load(src_ptr + src_row * src_stride + cols, mask=mask)
    tl.store(dst_ptr + dst_row * dst_stride + cols, values.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.jit
def fill_rows_kernel(
    dst_ptr,
    dst_stride,
    index_ptr,
    index_stride,
    count_ptr,
    value,
    width,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0)
    if lane >= tl.load(count_ptr):
        return
    row = tl.load(index_ptr + lane * index_stride).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < width
    values = tl.zeros([BLOCK], dtype=tl.float32) + value
    tl.store(dst_ptr + row * dst_stride + cols, values.to(dst_ptr.dtype.element_ty), mask=mask)


def as_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(1) if tensor.dim() == 1 else tensor


def block_width(width: int) -> int:
    return max(16, min(1024, triton.next_power_of_2(width)))


def copy_rows(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_index: torch.Tensor,
    src_index: torch.Tensor,
    count: torch.Tensor,
    lanes: int,
):
    dst, src = as_rows(dst), as_rows(src)
    width = dst.size(1)
    assert src.size(1) == width and dst.stride(1) == 1 and src.stride(1) == 1
    block = block_width(width)
    copy_rows_kernel[(lanes, triton.cdiv(width, block))](
        dst, dst.stride(0), src, src.stride(0),
        dst_index, dst_index.stride(0), src_index, src_index.stride(0),
        count, width, BLOCK=block,
    )


def fill_rows(
    dst: torch.Tensor,
    index: torch.Tensor,
    count: torch.Tensor,
    value: float,
    lanes: int,
):
    dst = as_rows(dst)
    width = dst.size(1)
    assert dst.stride(1) == 1
    block = block_width(width)
    fill_rows_kernel[(lanes, triton.cdiv(width, block))](
        dst, dst.stride(0), index, index.stride(0), count, float(value), width, BLOCK=block,
    )
