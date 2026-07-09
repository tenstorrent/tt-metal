# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Data-movement example: where you PLACE the reader cores changes DRAM NoC contention.

`reader_placement` does a pure interleaved DRAM->DRAM identity copy across a line of
`num_cores` Tensix cores. Every core reads its contiguous slice of pages on NoC0 and
writes them straight back on NoC1. The reader/writer kernels are byte-identical for
every placement — the ONLY thing that changes is which physical cores run them:

    placement="column"   : the left column   (0,0)..(0,N-1)  -- the split_work_to_cores
                            row_wise=False default; the trap / baseline.
    placement="row"      : the top row        (0,0)..(N-1,0)
    placement="diagonal" : the diagonal       (i,i) for i in 0..N-1

Because the work is identical, any measured difference is attributable purely to the
NoC routes those placements take to/from DRAM. See README.md for the mechanism, the
numbers, and how to run it on your own shapes.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_ID = 0  # reader -> writer

# Ordered so the "trap" is the baseline (first): column is what split_work_to_cores
# produces by default (row_wise=False), then the two better placements.
PLACEMENTS = ("column", "row", "diagonal")


def validate(input_tensor):
    """Keep the example dead simple: 2D, TILE, bf16, tile-aligned, interleaved."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"reader_placement example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("reader_placement example: input must be TILE_LAYOUT")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("reader_placement example: input must be bfloat16")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"reader_placement example: H and W must be multiples of {TILE}, got {shape}")


def _ordered_cores(placement, n):
    """The N cores of a placement, in a fixed order so page assignment is identical
    across placements (core index k always gets the same page range; only its
    physical (x, y) differs)."""
    if placement == "row":
        return [ttnn.CoreCoord(i, 0) for i in range(n)]
    if placement == "column":
        return [ttnn.CoreCoord(0, i) for i in range(n)]
    if placement == "diagonal":
        return [ttnn.CoreCoord(i, i) for i in range(n)]
    raise ValueError(f"reader_placement example: placement must be one of {PLACEMENTS}, got {placement!r}")


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_pages(num_pages, n):
    """Contiguous page ranges by core index; remainder spread over the first cores.
    Identical for every placement -> identical per-core work."""
    base, rem = divmod(num_pages, n)
    ranges = []
    start = 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def _resolve_num_cores(device, num_cores):
    """Cap at min(grid.x, grid.y) so a diagonal of this length fits on the grid."""
    grid = device.compute_with_storage_grid_size()
    max_line = min(grid.x, grid.y)
    if num_cores is None:
        return max_line
    if num_cores < 1 or num_cores > max_line:
        raise ValueError(
            f"reader_placement example: num_cores must be in [1, {max_line}] "
            f"(a diagonal must fit on the {grid.x}x{grid.y} grid), got {num_cores}"
        )
    return num_cores


def _create_program_descriptor(input_tensor, output_tensor, *, placement, num_cores, kernel_iters, block):
    device = input_tensor.device()
    num_cores = _resolve_num_cores(device, num_cores)

    page_bytes = input_tensor.buffer_aligned_page_size()  # one bf16 tile = 2048 B
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()
    if num_pages < num_cores:
        raise ValueError(f"reader_placement example: need >= {num_cores} pages, got {num_pages}")

    cores = _ordered_cores(placement, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_pages(num_pages, num_cores)

    cb = ttnn.CBDescriptor(
        # Two blocks deep: `block` reads can be in flight while the writer drains the
        # other block. Still a bounded, page-count-independent L1 footprint.
        total_size=2 * block * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_ID, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, kernel_iters, block]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [page_bytes, kernel_iters, block]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        reader_rt_args[core.x][core.y] = [in_addr, start_page, count]
        writer_rt_args[core.x][core.y] = [out_addr, start_page, count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "copy_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "copy_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb])


def reader_placement(
    input_tensor: ttnn.Tensor,
    *,
    placement: str = "row",
    num_cores: int = None,
    kernel_iters: int = 1,
    block: int = 16,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Interleaved DRAM->DRAM identity copy over a line of cores placed as `placement`.

    Args:
        placement: "column" (left column, the split_work_to_cores default trap),
            "row" (top row), or "diagonal" ((i,i)). Same kernel; only the physical
            cores differ.
        num_cores: line length. Defaults to min(grid.x, grid.y) so the diagonal fits.
        kernel_iters: in-kernel repeat of the read/write range. 1 = latency,
            large = steady-state throughput.
        block: pages issued per barrier (outstanding transactions). Larger = more
            NoC pressure = contention shows more clearly. Must be >= 1.

    Output is identical for every placement (== input).
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"reader_placement example: placement must be one of {PLACEMENTS}, got {placement!r}")
    if block < 1:
        raise ValueError(f"reader_placement example: block must be >= 1, got {block}")
    validate(input_tensor)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    program_descriptor = _create_program_descriptor(
        input_tensor, output_tensor, placement=placement, num_cores=num_cores, kernel_iters=kernel_iters, block=block
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
