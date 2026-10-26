# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wide reads for contiguous DRAM shards, with optional bank-affine work assignment."""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

BLOCK_ROWS = 32
CB_DATA = 0  # reader -> writer: unchanged ROW_MAJOR bytes
CB_DEPTH = 2  # blocks per CB; held constant across variants

VARIANTS = {
    "none": dict(bank_affinity=False, coalesce_reads=False),
    "coalesce": dict(bank_affinity=False, coalesce_reads=True),
    "affinity": dict(bank_affinity=True, coalesce_reads=False),
    "both": dict(bank_affinity=True, coalesce_reads=True),
}


def validate(input_tensor, shard_width):
    shape = list(input_tensor.shape)
    if len(shape) != 2 or min(shape) <= 0 or shape[0] % BLOCK_ROWS:
        raise ValueError("read_coalescing requires positive 2D input with H divisible by 32")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT or input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("read_coalescing requires ROW_MAJOR bf16 input")
    mc = input_tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise ValueError("read_coalescing requires width-sharded DRAM input")
    nb = num_dram_banks(input_tensor.device())
    if shard_width not in (32, 64, 128, 256, 512) or shape[1] != nb * shard_width:
        raise ValueError("one full-width chunk per bank is required (shard_width in {32, 64, 128, 256, 512})")
    spec = mc.shard_spec
    if (
        list(spec.shape) != [shape[0], shard_width]
        or spec.grid.num_cores() != nb
        or spec.orientation != ttnn.ShardOrientation.ROW_MAJOR
    ):
        raise ValueError("requires one full-height shard per bank, row-major orientation")
    if input_tensor.buffer_page_size() != shard_width * 2 or input_tensor.buffer_aligned_page_size() != shard_width * 2:
        raise ValueError("source pages must be full chunk rows without alignment gaps")


def work_geometry(shape, shard_width, grid_cores):
    h, w = shape
    nt_h, n_w = h // BLOCK_ROWS, w // (shard_width)
    # Whole bank groups keep the active core set identical across all four arms.
    groups = min(grid_cores // n_w, nt_h)
    if groups < 1:
        raise ValueError("the compute grid must have at least one core per bank")
    return nt_h, n_w, nt_h * n_w, groups * n_w


def _ordered_cores(device, n):
    """`n` cores filled row-major across the grid -- identical for every variant."""
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _split_units(units, n):
    """Contiguous unit ranges by core index; remainder on the first cores."""
    base, rem = divmod(units, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def num_dram_banks(device):
    return ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks


def create_program_descriptor(
    input_tensor, output_tensor, *, bank_affinity, coalesce_reads, shard_width, kernel_iters=1
):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    h, w = list(input_tensor.shape)
    _, n_w, units, num_cores = work_geometry((h, w), shard_width, grid.x * grid.y)

    page_bytes = input_tensor.buffer_aligned_page_size()  # one row, or one shard-row if width-sharded
    pages_per_row = w * 2 // page_bytes
    chunk_row_bytes = shard_width * 2  # bf16 slice of one row
    output_row_bytes = output_tensor.buffer_aligned_page_size()
    block_bytes = BLOCK_ROWS * chunk_row_bytes

    cores = _ordered_cores(device, num_cores)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    assignment = _split_units(units, num_cores)
    if bank_affinity:
        rows = _split_units(h // BLOCK_ROWS, num_cores // n_w)
        assignment = [(rows[k // n_w][0] * n_w + k % n_w, rows[k // n_w][1]) for k in range(num_cores)]

    def cb(index, dtype):
        return ttnn.CBDescriptor(
            total_size=CB_DEPTH * block_bytes,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=block_bytes)],
        )

    reader_ct = [
        BLOCK_ROWS,
        page_bytes,
        chunk_row_bytes,
        kernel_iters,
        int(bank_affinity),
        int(coalesce_reads),
        pages_per_row,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [BLOCK_ROWS, output_row_bytes, kernel_iters, int(bank_affinity), chunk_row_bytes]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for core, (start, count) in zip(cores, assignment):
        reader_rt[core.x][core.y] = [in_addr, start, count, n_w]
        writer_rt[core.x][core.y] = [out_addr, start, count, n_w]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb(CB_DATA, input_tensor.dtype)])


def read_coalescing(
    input_tensor: ttnn.Tensor,
    *,
    bank_affinity: bool = False,
    coalesce_reads: bool = True,
    shard_width: int = 256,
    kernel_iters: int = 1,
) -> ttnn.Tensor:
    """Copy width-sharded DRAM to interleaved DRAM, preserving ROW_MAJOR layout and bytes."""
    if kernel_iters < 1:
        raise ValueError(f"read_coalescing example: kernel_iters must be >= 1, got {kernel_iters}")
    validate(input_tensor, shard_width)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        input_tensor.device(),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        bank_affinity=bank_affinity,
        coalesce_reads=coalesce_reads,
        shard_width=shard_width,
        kernel_iters=kernel_iters,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
