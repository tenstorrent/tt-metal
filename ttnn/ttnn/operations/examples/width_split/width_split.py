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

A second pair, `width_split_reduce()` (row_split_reduce vs width_split_reduce), covers the case
the elementwise op hides: a per-row reduction + per-row apply, where a width split makes the k
Tensix cores that share a tile-row pay one combine round per row. See the bottom of this file
and README.md's "Reduce case". The same pair also accepts a ROW_MAJOR input (one DRAM page = one
stick of W datums): there, splitting the row also splits every stick into k smaller DRAM transfers.
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

# Reduce case (per-row reduction + per-row apply, so a width split needs a cross-core combine).
# Baseline first: row_split_reduce = whole tile-rows per Tensix core, no combine.
REDUCE_VARIANTS = ("row_split_reduce", "width_split_reduce")

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


# ---------------------------------------------------------------------------
# Reduce case: y = x - mean_W(x) per row (one per-row reduction + one per-row apply).
#
# A width split of a row-reduction is not free: the k Tensix cores that share a tile-row each
# hold only a partial, so every row pays a combine round between them before any of them can
# apply the result. This pair isolates exactly that trade:
#
#   variant="row_split_reduce"   : whole tile-rows per Tensix core over min(R_t, grid) Tensix
#                                  cores (the honest baseline: no combine at all).
#   variant="width_split_reduce" : each tile-row split into k slices of Wt/k tiles over a group
#                                  of k Tensix cores; min(R_t, grid // k) groups. The k members
#                                  exchange their 1-tile partials (symmetric all-gather, one
#                                  round, see kernels/ws_reduce_writer.cpp) and each subtracts
#                                  the same mean from its own slice.
#
# Same three kernels for both; k is a runtime arg (k = 1 skips the exchange).
#
# ROW_MAJOR input (Layout::ROW_MAJOR, one DRAM page = one stick of W bf16 datums = 64 * Wt B): same
# variants, same combine; the kernels are compiled with WS_ROW_MAJOR. Per tile-row each Tensix core
# reads its piece of each of the 32 sticks (32 reads of 64 * Wt / k B), tilizes, reduces / applies,
# untilizes and writes 32 pieces back. A Tensix core issues 32 reads + 32 writes per tile-row
# whatever k is, so a width split multiplies the grid's DRAM transfer count by k and divides each
# transfer's size by k: that is the extra cost of splitting a stick.
# ---------------------------------------------------------------------------

CB_SCALER = 1
CB_SEND = 2
CB_GATHER = 3
CB_STAT = 4
CB_RM_IN = 5  # ROW_MAJOR only: 32 stick pieces of the slice, before tilize
CB_Y = 6  # ROW_MAJOR only: tiled result, before untilize
SEM_ARRIVED = 0
TILE_BYTES = TILE * TILE * 2  # bf16 tile

# Tiles per NoC barrier / CB push / DST block: one half-sync DST bank with fp32 accumulation.
REDUCE_BLOCK = 4


def validate_reduce(input_tensor, k):
    shape = list(input_tensor.shape)
    if len(shape) != 2:
        raise ValueError(f"width_split reduce: rank must be 2, got {len(shape)}")
    if input_tensor.layout not in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        raise ValueError("width_split reduce: input must be TILE_LAYOUT or ROW_MAJOR_LAYOUT")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError(f"width_split reduce: dtype must be bfloat16, got {input_tensor.dtype}")
    h, w = shape
    if h % TILE or w % TILE:
        raise ValueError(f"width_split reduce: H and W must be multiples of {TILE}, got {shape}")
    wt = w // TILE
    if k < 1 or wt % k:
        raise ValueError(f"width_split reduce: split factor k={k} must divide Wt={wt}")
    if k > _grid_cores(input_tensor.device()):
        raise ValueError(f"width_split reduce: k={k} exceeds the Tensix grid")
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT and 6 * (wt // k) * TILE_BYTES > 900 * 1024:
        raise ValueError(f"width_split reduce: ROW_MAJOR slice of {wt // k} tiles does not fit L1 (6 slices of CBs)")


def reduce_layout(device, rows_t, wt, variant, k):
    """(groups, k, slice tiles, per-group (row_start, num_rows)). Group g = Tensix cores
    g*k .. g*k + k - 1 in row-major grid order (members of a group are neighbours)."""
    if variant == "row_split_reduce":
        k = 1
    groups = max(1, min(rows_t, _grid_cores(device) // k))
    return groups, k, wt // k, _assign_pages(rows_t, groups)


def reduce_active_cores(device, rows_t, wt, variant, k):
    groups, k, _, _ = reduce_layout(device, rows_t, wt, variant, k)
    return groups * k


def create_reduce_program_descriptor(input_tensor, output_tensor, *, variant, k=2, kernel_iters=1):
    if variant not in REDUCE_VARIANTS:
        raise ValueError(f"width_split reduce: variant must be one of {REDUCE_VARIANTS}, got {variant!r}")
    device = input_tensor.device()
    h, w = list(input_tensor.shape)
    rows_t, wt = h // TILE, w // TILE
    groups, k, slice_tiles, row_ranges = reduce_layout(device, rows_t, wt, variant, k)
    blk = min(REDUCE_BLOCK, slice_tiles)
    if slice_tiles % blk:
        raise ValueError(f"width_split reduce: slice of {slice_tiles} tiles must be a multiple of {blk}")

    # page_bytes = one DRAM page: a tile (TILE) or one stick of W bf16 datums (ROW_MAJOR). CB pages
    # are always tiles: the ROW_MAJOR path stages 32 stick pieces as S tile-sized pages.
    page_bytes = input_tensor.buffer_aligned_page_size()
    row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    cores = _ordered_cores(device, groups * k)
    core_ranges = _core_range_set(cores)

    def cb(index, tiles):
        return ttnn.CBDescriptor(
            total_size=tiles * TILE_BYTES,
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE_BYTES)
            ],
        )

    # cb_in holds two whole slices: the slice stays resident from reduce to apply, and the reader
    # can prefetch the next tile-row's slice meanwhile. cb_gather: 2 rows x k partial slots.
    # ROW_MAJOR: the reader prefetches into cb_rm_in (2 slices) instead, compute tilizes one slice
    # into cb_in, applies into cb_y and untilizes into cb_out (2 slices, so the writer drains one
    # tile-row's sticks while compute produces the next).
    if row_major:
        cbs = [
            cb(CB_RM_IN, 2 * slice_tiles),
            cb(CB_IN, slice_tiles),
            cb(CB_SCALER, 1),
            cb(CB_SEND, 1),
            cb(CB_GATHER, 2 * k),
            cb(CB_STAT, 1),
            cb(CB_Y, slice_tiles),
            cb(CB_OUT, 2 * slice_tiles),
        ]
        defines = [("WS_ROW_MAJOR", "1"), ("WS_RM_SLICE", str(slice_tiles))]
    else:
        cbs = [
            cb(CB_IN, 2 * slice_tiles),
            cb(CB_SCALER, 1),
            cb(CB_SEND, 1),
            cb(CB_GATHER, 2 * k),
            cb(CB_STAT, 1),
            cb(CB_OUT, 2 * blk),
        ]
        defines = []
    # ROW_MAJOR: each Tensix core owns slice_tiles * 32 datums of every stick = slice_tiles * 64 B,
    # starting slice_start * 64 B into the stick (always a multiple of the 64 B DRAM alignment).
    slice_bytes = slice_tiles * TILE * 2

    reader_ct = [page_bytes, kernel_iters] + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = [page_bytes, kernel_iters, SEM_ARRIVED] + list(
        ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    )
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    in_addr, out_addr = input_tensor.buffer_address(), output_tensor.buffer_address()
    for g, (row_start, num_rows) in enumerate(row_ranges):
        members = cores[g * k : (g + 1) * k]
        noc_xy = []
        for c in members:
            v = device.worker_core_from_logical_core(c)
            noc_xy.extend([v.x, v.y])
        for m, c in enumerate(members):
            slice_start = m * slice_tiles
            rm_args = [slice_bytes, slice_start * TILE * 2] if row_major else []
            reader_rt[c.x][c.y] = [in_addr, wt, slice_tiles, slice_start, row_start, num_rows, blk] + rm_args
            writer_rt[c.x][c.y] = [out_addr, wt, slice_tiles, slice_start, row_start, num_rows, blk, k, m] + noc_xy
            writer_rt[c.x][c.y] = list(writer_rt[c.x][c.y]) + rm_args
            compute_rt[c.x][c.y] = [slice_tiles, num_rows, k, blk]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ws_reduce_reader.cpp"),
        core_ranges=core_ranges,
        defines=defines,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # reader -> NoC0
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ws_reduce_writer.cpp"),
        core_ranges=core_ranges,
        defines=defines,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # writer + combine -> NoC1
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "ws_reduce_compute.cpp"),
        core_ranges=core_ranges,
        defines=defines,
        compile_time_args=[kernel_iters],
        runtime_args=compute_rt,
        # HiFi4 + fp32 DST, identical for both variants: a row-sum accumulated across Wt tiles in a
        # bf16 DST loses the small per-tile increments (Wt = 64 tiles fails PCC 0.999 without it).
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_ARRIVED, core_ranges=core_ranges, initial_value=0)]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)


def width_split_reduce(
    input_tensor: ttnn.Tensor,
    *,
    variant: str = "width_split_reduce",
    k: int = 2,
    kernel_iters: int = 1,
) -> ttnn.Tensor:
    """Per-row centering y = x - mean_W(x) over an interleaved DRAM bf16 tensor [32*R_t, 32*Wt].

    The input may be TILE_LAYOUT (pages are tiles) or ROW_MAJOR_LAYOUT (pages are sticks of W datums;
    the kernels read stick pieces, tilize, reduce/apply, untilize and write stick pieces back). The
    output has the input's layout.

    Args:
        variant: "row_split_reduce" (whole tile-rows per Tensix core, no combine — baseline) or
            "width_split_reduce" (each tile-row over a group of k Tensix cores + a per-row combine).
        k: split factor for width_split_reduce (must divide Wt; ignored by row_split_reduce).
        kernel_iters: in-kernel repeat. 1 = per-launch latency.
    """
    if kernel_iters < 1:
        raise ValueError(f"width_split reduce: kernel_iters must be >= 1, got {kernel_iters}")
    if variant == "row_split_reduce":
        k = 1
    validate_reduce(input_tensor, k)
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), ttnn.bfloat16, input_tensor.layout, device, ttnn.DRAM_MEMORY_CONFIG
    )
    descriptor = create_reduce_program_descriptor(
        input_tensor, output_tensor, variant=variant, k=k, kernel_iters=kernel_iters
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)
