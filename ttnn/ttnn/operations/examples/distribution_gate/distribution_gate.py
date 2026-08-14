# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Work-distribution example: GATING which axis you split, so a fix for one
aspect-ratio regime does not regress the other.

To spread a tile op over the grid you split the tiles across cores along SOME
axis. Two natural choices:

    "height_split" : partition the tile-ROWS across cores (each core owns a
                     contiguous band of rows, full width). Fills the grid when
                     there are many tile-rows; strands a WIDE-SHORT tensor
                     (few rows, many columns) on as few as ONE core.
    "width_split"  : partition the tile-COLUMNS across cores (each core owns a
                     column range, full height). Fills the grid when there are
                     many tile-columns; strands a TALL-NARROW tensor (many rows,
                     few columns) on as few as ONE core.

Each is the RIGHT choice for one regime and a disaster for the other — the trap
is symmetric. The tempting "fix" for a wide-short tensor is to switch wholesale
to width_split; but that switch REGRESSES every tall-narrow tensor the height
split already handled. The disciplined fix is a "gated" strategy: keep the
conventional height split as the DEFAULT, and divert to width_split ONLY when the
height split would leave the grid under-filled. When the gate does not trip, the
default path is untouched — so the shapes it already handled cannot regress.

Compute is a trivial per-tile op (one relu) on interleaved DRAM tile tensors, and
all three variants share BYTE-IDENTICAL kernels; only the per-core tile rectangle
(and therefore how many cores run) changes. So the measured delta is purely WORK
DISTRIBUTION. Sweep the aspect ratio (see README.md) to watch each fixed split
collapse on its bad regime while the gate fills the grid on both.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB_IN = 0  # reader -> compute
CB_OUT = 16  # compute -> writer

# Tiles a core streams through its CB per NoC barrier (bounds per-core L1, and the
# reads/writes issued per barrier). Held constant across variants so the only
# thing that varies is which tiles — and how many cores — the work is split over.
BLOCK = 8

# height_split / width_split are the two fixed single-axis strategies; gated is the
# disciplined divert. Height is listed first: it is the conventional default the
# gate preserves.
VARIANTS = ("height_split", "width_split", "gated")

SUPPORTED_DTYPES = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b)


def validate(input_tensor):
    """2D, TILE layout, tile-aligned in both dims, interleaved; dtype supported.
    Unlike a wide-short-only demo, H is free — we sweep the aspect ratio."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"distribution_gate example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("distribution_gate example: input must be TILE_LAYOUT")
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"distribution_gate example: dtype must be one of {SUPPORTED_DTYPES}, got {input_tensor.dtype}"
        )
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"distribution_gate example: H and W must be multiples of {TILE}, got ({h}, {w})")


def _grid_cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _ordered_cores(device, n):
    """`n` cores filled row-major — identical placement for every variant, so only
    the *count* of active cores (and which tiles they own) varies."""
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _split_contiguous(total, n):
    """Partition `total` units into `n` contiguous (start, count) ranges;
    remainder lands on the first cores. Empty ranges are dropped by the caller."""
    base, rem = divmod(total, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        if count:
            ranges.append((start, count))
        start += count
    return ranges


def plan(variant, ht, wt, grid_cores):
    """Return (chosen_axis, rectangles) where each rectangle is
    (row_start, row_count, col_start, col_count) — the tile block one core owns.

    height_split -> split rows, full-width bands   (cores = min(ht, grid))
    width_split  -> split cols, full-height strips  (cores = min(wt, grid))
    gated        -> height by default; divert to width ONLY when width strictly
                    fills more cores than height would (i.e. height under-fills
                    and width does better). Ties keep the default height path, so
                    a shape the height split already saturated is never perturbed.
    """
    height_cores = min(ht, grid_cores)
    width_cores = min(wt, grid_cores)

    if variant == "gated":
        chosen = "width_split" if width_cores > height_cores else "height_split"
    elif variant in ("height_split", "width_split"):
        chosen = variant
    else:
        raise ValueError(f"distribution_gate example: variant must be one of {VARIANTS}, got {variant!r}")

    if chosen == "height_split":
        rects = [(r0, rc, 0, wt) for (r0, rc) in _split_contiguous(ht, height_cores)]
    else:
        rects = [(0, ht, c0, cc) for (c0, cc) in _split_contiguous(wt, width_cores)]
    return chosen, rects


def num_active_cores(variant, device, ht, wt):
    """How many cores `variant` actually engages for this aspect ratio."""
    _, rects = plan(variant, ht, wt, _grid_cores(device))
    return len(rects)


def create_program_descriptor(input_tensor, output_tensor, *, variant, kernel_iters=1, block=BLOCK):
    if variant not in VARIANTS:
        raise ValueError(f"distribution_gate example: variant must be one of {VARIANTS}, got {variant!r}")
    device = input_tensor.device()

    page_bytes = input_tensor.buffer_aligned_page_size()
    h, w = list(input_tensor.shape)
    ht, wt = h // TILE, w // TILE

    _, rects = plan(variant, ht, wt, _grid_cores(device))
    num_cores = len(rects)
    cores = _ordered_cores(device, num_cores)
    core_ranges = _core_range_set(cores)

    # Double-buffered CB held constant across variants (2 * block tiles) so the ONLY
    # variable is which tiles — and how many cores — the work is spread over.
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
    for core, (row_start, row_count, col_start, col_count) in zip(cores, rects):
        # reader/writer walk this core's (row_count x col_count) tile rectangle in
        # row-major order; page(r, c) = r * Wt + c (contiguous for a full-width band,
        # strided for a column strip — the kernel handles both identically).
        args = [row_start, row_count, col_start, col_count, wt]
        reader_rt[core.x][core.y] = [in_addr, *args]
        writer_rt[core.x][core.y] = [out_addr, *args]
        compute_rt[core.x][core.y] = [row_count * col_count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "dg_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "dg_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # writer -> NoC1
    )
    fp32_dest_acc = input_tensor.dtype == ttnn.float32
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "dg_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=[cb_in, cb_out]
    )


def distribution_gate(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "gated",
    kernel_iters: int = 1,
    block: int = BLOCK,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """relu over an interleaved DRAM tile tensor, distributed three ways.

    Args:
        variant: "height_split" (split tile-rows — strands wide-short tensors),
            "width_split" (split tile-columns — strands tall-narrow tensors), or
            "gated" (height by default, divert to width only when height under-fills
            the grid — fills the grid on both regimes without regressing either).
            Same kernels; only the per-core tile rectangle / active-core count differ.
        kernel_iters: in-kernel repeat of the tile range. 1 = per-launch latency,
            large = steady-state throughput.
        block: tiles per NoC barrier / CB block (bounds per-core L1); BLOCK default.

    Output is relu(input) for all three variants.
    """
    if kernel_iters < 1:
        raise ValueError(f"distribution_gate example: kernel_iters must be >= 1, got {kernel_iters}")
    if block < 1:
        raise ValueError(f"distribution_gate example: block must be >= 1, got {block}")
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
