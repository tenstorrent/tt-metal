# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""How many cores? — DRAM-bandwidth SATURATION vs core count on a DRAM-bound op.

For a DRAM-bound op (the movement, not the math, is the whole job) there is a
number-of-cores decision that is easy to get wrong: "more cores = faster" holds
only until the DRAM interface saturates. Past that knee, extra cores add **no**
bandwidth — they just pile more traffic onto the same banks / NoC links, so they
are wasted, and if they are placed badly they *congest* and the op gets SLOWER.
The right answer for a bandwidth-bound op is the *minimum* well-placed cores that
saturate the bus — not the whole grid.

This example is a pure DRAM→DRAM copy (read on NoC0, write on NoC1, no compute) of
a fixed large interleaved tensor, and it **sweeps the core count**. Achieved
bandwidth is `2 * tensor_bytes / device_kernel_duration` (read + write). Two
placements make the mechanism visible:

    variant="spread"  : the N cores are placed row-major across the grid, so their
                        traffic spreads over the DRAM-facing axis. Bandwidth rises
                        with cores, then PLATEAUS at ~DRAM peak — the plateau onset
                        is the sweet-spot core count.
    variant="stacked" : the N cores are stacked column-major (piled onto one axis),
                        so they share NoC links. Bandwidth saturates LOWER and can
                        ROLL OVER — more cores make it slower.

The reader/writer kernels are BYTE-IDENTICAL across variants and core counts; only
which cores run and how the tiles are assigned changes. So the measured delta is
purely work distribution + placement, and the sweep answers "how many cores should
a DRAM-bound op use, and does adding more keep paying?" See README.md.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
CB = 0  # reader -> writer (single CB; no compute — this is a pure copy)

# Tiles a core streams per NoC barrier (bounds per-core L1 and reads/writes issued
# per barrier). Held constant across variants and core counts, double-buffered, so
# the only thing that varies is how many cores run and where they sit.
BLOCK = 4

# spread first (the good placement); stacked shows the congestion rollover.
VARIANTS = ("spread", "stacked")

SUPPORTED_DTYPES = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b)


def validate(input_tensor):
    """2D, TILE layout, tile-aligned, interleaved DRAM, supported dtype. The tensor
    is deliberately large so the copy is DRAM-bandwidth-bound (the regime where the
    core-count knee exists)."""
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"dram_saturation example: rank must be 2, got {len(shape)}")
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("dram_saturation example: input must be TILE_LAYOUT")
    if input_tensor.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"dram_saturation example: dtype must be one of {SUPPORTED_DTYPES}, got {input_tensor.dtype}")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"dram_saturation example: H and W must be multiples of {TILE}, got ({h}, {w})")


def _grid_cores(device):
    grid = device.compute_with_storage_grid_size()
    return grid.x * grid.y


def _placed_cores(device, variant, n):
    """`n` cores, placed by variant:
      spread  -> row-major   (k -> (k % gx, k // gx)): the line spreads across the
                 x-axis first, dispersing traffic over the DRAM-facing axis.
      stacked -> column-major (k -> (k // gy, k % gy)): the line piles down one
                 column first, concentrating traffic on shared NoC links.
    Same core COUNT for both — only the geometry differs."""
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    if variant == "spread":
        return [ttnn.CoreCoord(k % gx, k // gx) for k in range(n)]
    elif variant == "stacked":
        return [ttnn.CoreCoord(k // gy, k % gy) for k in range(n)]
    raise ValueError(f"dram_saturation example: variant must be one of {VARIANTS}, got {variant!r}")


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _assign_pages(num_pages, n):
    """Contiguous page ranges by core index; remainder on the first cores."""
    base, rem = divmod(num_pages, n)
    ranges, start = [], 0
    for k in range(n):
        count = base + (1 if k < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


def num_active_cores(device, num_cores):
    """Resolve the requested core count against the grid (None -> full grid)."""
    grid_cores = _grid_cores(device)
    if num_cores is None:
        return grid_cores
    return max(1, min(int(num_cores), grid_cores))


def sweet_spot_cores(gbps_by_cores, tol=0.03):
    """Given a measured {core_count: achieved_GB/s} curve, return the **sweet spot**:
    the SMALLEST core count whose bandwidth is within `tol` of the peak. This is the
    exploit — a DRAM-bound op saturates here, so any cores beyond it add no bandwidth
    and are free to spend elsewhere. Pure analysis over a measured sweep; no device."""
    if not gbps_by_cores:
        raise ValueError("dram_saturation example: empty measurement curve")
    peak = max(gbps_by_cores.values())
    threshold = peak * (1.0 - tol)
    return min(c for c, g in gbps_by_cores.items() if g >= threshold)


def create_program_descriptor(input_tensor, output_tensor, *, variant, num_cores=None, kernel_iters=1, block=BLOCK):
    if variant not in VARIANTS:
        raise ValueError(f"dram_saturation example: variant must be one of {VARIANTS}, got {variant!r}")
    device = input_tensor.device()

    page_bytes = input_tensor.buffer_aligned_page_size()
    num_pages = output_tensor.buffer_num_pages()
    assert num_pages == input_tensor.buffer_num_pages()

    n = num_active_cores(device, num_cores)
    # Never launch more cores than there are tiles to move.
    n = max(1, min(n, num_pages))
    cores = _placed_cores(device, variant, n)
    core_ranges = _core_range_set(cores)
    assignment = _assign_pages(num_pages, n)

    # Double-buffered single CB (reader -> writer). Constant across variants/counts.
    cb_tiles = 2 * block
    cb = ttnn.CBDescriptor(
        total_size=cb_tiles * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB, data_format=input_tensor.dtype, page_size=page_bytes)
        ],
    )

    reader_ct_args = [page_bytes, kernel_iters, block]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [page_bytes, kernel_iters, block]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, (start_page, count) in zip(cores, assignment):
        reader_rt[core.x][core.y] = [in_addr, start_page, count]
        writer_rt[core.x][core.y] = [out_addr, start_page, count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ds_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ds_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel], semaphores=[], cbs=[cb])


def dram_saturation(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "spread",
    num_cores: int = None,
    kernel_iters: int = 1,
    block: int = BLOCK,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """Identity DRAM→DRAM copy, distributed across `num_cores` cores (placed by
    `variant`). The output equals the input; the point is the *bandwidth* the copy
    achieves as a function of core count and placement.

    Args:
        variant: "spread" (cores row-major across the grid — traffic spread over the
            DRAM-facing axis) or "stacked" (cores column-major — piled onto shared
            NoC links). Same count, different geometry.
        num_cores: how many cores run the copy (None = full grid). This is the axis
            to sweep — watch achieved bandwidth saturate.
        kernel_iters: in-kernel repeat. 1 = per-launch latency; large = steady-state.
        block: tiles per NoC barrier / CB block (double-buffered); BLOCK default.
    """
    if kernel_iters < 1:
        raise ValueError(f"dram_saturation example: kernel_iters must be >= 1, got {kernel_iters}")
    if block < 1:
        raise ValueError(f"dram_saturation example: block must be >= 1, got {block}")
    validate(input_tensor)
    device = input_tensor.device()
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, device, out_mem
    )
    program_descriptor = create_program_descriptor(
        input_tensor, output_tensor, variant=variant, num_cores=num_cores, kernel_iters=kernel_iters, block=block
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
