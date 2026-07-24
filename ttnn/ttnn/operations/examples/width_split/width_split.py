# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Work-distribution example: WIDTH-SPLITTING a wide, short tensor across the grid.

For a wide, SHORT tensor (few tile-rows, many tile-columns), the natural "split
the work by tile-rows" strategy strands everything on `nt_h` cores — as few as
ONE when the tensor is a single tile-row tall. The rest of the compute grid sits
idle and the op runs at single-core speed no matter how wide the tensor is. The
fix is a WIDTH split: hand each core a contiguous range of tile-COLUMNS (bounded
by a `WT_CHUNK` constant so per-core L1 stays constant), so the whole grid works
in parallel.

This example fixes the tensor at ONE tile-row tall (H=32) and isolates that one
decision on a trivial per-tile op (relu over an interleaved DRAM tile tensor), so
the measured delta is purely WORK DISTRIBUTION — not compute:

    variant="single_core" : all Wt tiles on ONE core (what a tile-row split
                            degenerates to for a 1-tile-row-tall tensor).
    variant="width_split" : the Wt tiles spread contiguously across
                            min(Wt, grid_cores) cores.

The reader / compute / writer kernels are BYTE-IDENTICAL for both variants — only
the per-core (start_page, num_pages) runtime args (and hence how many cores run)
change. Sweep the width to see width_split go from no benefit (tiny Wt, ~1 core)
to ~grid-x faster (Wt >> grid). See README.md for the numbers and the crossover.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_IN = 0  # reader -> compute
CB_OUT = 16  # compute -> writer

# Max tile-columns a core holds in its CB at once (bounds per-core L1, and the
# reads/writes issued per NoC barrier). Held constant across variants so the only
# thing that changes is how many cores the Wt tiles are spread over.
WT_CHUNK = 8

# Baseline first: single_core is the trap for a wide-short tensor; width_split fills the grid.
VARIANTS = ("single_core", "width_split")

SUPPORTED_DTYPES = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b)


def validate(input_tensor):
    """Dead simple: 2D, TILE, ONE tile-row tall (H=32, the wide-short case),
    tile-aligned width, interleaved; dtype in SUPPORTED_DTYPES."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"width_split example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("width_split example: input must be TILE_LAYOUT")
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"width_split example: dtype must be one of {SUPPORTED_DTYPES}, got {input_tensor.dtype}")
    h, w = shape
    if h != TILE:
        raise ValueError(f"width_split example: H must be exactly {TILE} (one tile-row, the wide-short case), got {h}")
    if w % TILE:
        raise ValueError(f"width_split example: W must be a multiple of {TILE}, got {w}")


def _grid_cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _ordered_cores(device, n):
    """`n` cores filled row-major across the grid — identical placement for both
    variants, so only the *count* of active cores varies."""
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_pages(num_pages, n):
    """Contiguous page ranges by core index; remainder on the first cores.
    For a 1-tile-row tensor, a page == a tile-column, so this IS the width split."""
    base, rem = divmod(num_pages, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def _num_cores_for(variant, device, num_pages):
    if variant == "single_core":
        return 1
    # width_split: fill the grid, but never more cores than there are tile-columns.
    return max(1, min(num_pages, _grid_cores(device)))


def create_program_descriptor(input_tensor, output_tensor, *, variant, kernel_iters=1, block=WT_CHUNK):
    if variant not in VARIANTS:
        raise ValueError(f"width_split example: variant must be one of {VARIANTS}, got {variant!r}")
    device = input_tensor.device()

    page_bytes = input_tensor.buffer_aligned_page_size()
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()

    num_cores = _num_cores_for(variant, device, num_pages)
    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)
    assignment = _assign_pages(num_pages, num_cores)

    # Double-buffered CB held constant across variants (2 * block tiles) so the ONLY
    # variable is how many cores share the Wt tiles.
    cb_tiles = 2 * block
    cb_in = ttnn.CBDescriptor(
        total_size=cb_tiles * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )
    cb_out = ttnn.CBDescriptor(
        total_size=cb_tiles * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=output_tensor.dtype, page_size=page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, kernel_iters, block]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [page_bytes, kernel_iters, block]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_ct_args = [kernel_iters]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        reader_rt[core.x][core.y] = [in_addr, start_page, count]
        writer_rt[core.x][core.y] = [out_addr, start_page, count]
        compute_rt[core.x][core.y] = [count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ws_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ws_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )
    fp32_dest_acc = input_tensor.dtype == ttnn.float32
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ws_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=[cb_in, cb_out]
    )


def width_split(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "width_split",
    kernel_iters: int = 1,
    block: int = WT_CHUNK,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """relu over a wide-short (H=32) interleaved DRAM tile tensor, distributed two ways.

    Args:
        variant: "single_core" (all Wt tiles on one core — the wide-short trap) or
            "width_split" (Wt tiles spread across min(Wt, grid) cores). Same kernels;
            only the per-core page assignment / active-core count differ.
        kernel_iters: in-kernel repeat of the tile range. 1 = per-launch latency,
            large = steady-state throughput.
        block: tile-columns per NoC barrier / CB block (bounds per-core L1); WT_CHUNK default.

    Output is relu(input) for both variants.
    """
    if kernel_iters < 1:
        raise ValueError(f"width_split example: kernel_iters must be >= 1, got {kernel_iters}")
    if block < 1:
        raise ValueError(f"width_split example: block must be >= 1, got {block}")
    validate(input_tensor)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        input_tensor, output_tensor, variant=variant, kernel_iters=kernel_iters, block=block
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
