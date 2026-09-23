# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor for the `row_split_interleaved`, `sharded_resident` and
`sharded_accessor` regimes (op_design.md -> Regimes).

Core assignment (`_core_assignment`, one source for all three regimes):
  * an L1-sharded OUTPUT whose shard is `resident_ok` fixes it (the output shard
    grid wins when both sides are sharded); else
  * an L1-sharded INPUT whose shard is `resident_ok`; else
  * `split_work_to_cores(grid, R, row_wise=True)` over the runtime grid.
Each side is then RESIDENT (its CB is backed on its own shard via
`ttnn.cb_descriptor_from_sharded_tensor`, zero NoC bytes) iff its per-core shard
rectangle equals the core's assigned rectangle; every other side is STREAMED
through `TensorAccessor` (interleaved DRAM/L1, DRAM-sharded, cross-spec remote
shards, ND specs without a 2-D equivalent).

Blocking knobs, each defined ONCE here and passed to the kernels as a single CT
or RT arg (every dependent quantity derives from it):

  tile_row  axis  ->  block_height = core_row_tiles   (RT, from split_work_to_cores)
                      rows_per_quantum = streaming window along tile_row (CT; from QUANTUM_MIN_TILES)
  tile_col  axis  ->  block_width  = balanced_width(core_col_tiles_max, block_width_cap)   (CT)
                      num_col_groups = 1 (Phase 0; grid_2d_split refinement turns it)
  depth knobs     ->  DEPTH_IN, DEPTH_OUT   (CB total_size and per_col_tile_bytes only)
  L1 budget       ->  CB_BUDGET_BYTES[low_l1]

Per Tensix core the kernels process the output-tile rectangle
[row_start, row_start + core_row_tiles) x [col_start, col_start + core_col_tiles),
cut into ceil(core_col_tiles / block_width) column blocks; the walk over them is
streamed rows_per_quantum tile-rows (rows_per_quantum * block_width tiles) per CB
quantum through two depth-2 CBs.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_WIDTH = 32  # elements per tile row (a tile's width is always 32)
FULL_TILE_HEIGHT = 32  # rows of a full (non-tiny) tile; tiny tiles are power-of-two fractions of it

# CB slots (semantic names; the index is just a slot).
CB_INPUT_STICKS = 0  # reader -> compute: tile_h stick segments per tile-row, block_width tile-sized pages
CB_OUTPUT_TILES = 1  # compute -> writer: block_width TILE pages per tile-row
CB_INPUT_STICKS_ODD = 2  # split reader only: BRISC -> compute, the odd tile-rows (one producer per CB)

# Buffer-depth knobs, counted in CB quanta (one quantum = rows_per_quantum tile-rows
# of block_width pages each).
DEPTH_IN = 2
DEPTH_OUT = 2

# CB quanta of stick reads a producer keeps in flight before it waits on the oldest
# one's (transaction-id) barrier (op_design.md perf lamp "Read in-flight depth").
# 1 = barrier every quantum before issuing the next. Must be <= DEPTH_IN.
# Measured flat on WH 64 cores (2 vs 1: 26.5 vs 26.9 us on [1,1,16384,64], 20.2 vs
# 19.8 us on [1,1,16384,32]): the pipeline is DRAM-traffic-bound there, not
# barrier-bound. Parked at the trivial value; the knob stays live.
READ_AHEAD = 1

# Minimum CB quantum in tiles along the tile_row streaming window. The reader's read
# barrier + CB push and the writer's flush + CB pop happen once per quantum of
# rows_per_quantum tile-rows (compute still tilizes one block_width tile-row per helper
# block). rows_per_quantum = ceil(QUANTUM_MIN_TILES / block_width), then capped by the
# busiest core's walk length // DEPTH_IN (so the double buffer still has quanta to
# overlap: one quantum per core serializes read -> tilize -> write) and by what the CB
# budget leaves after block_width is chosen. It never shrinks block_width, and it only
# engages where one tile-row is narrower than QUANTUM_MIN_TILES tiles. 1 = one
# handshake per tile-row everywhere.
# Measured on WH B0, 64 Tensix cores, bf16 (median device-kernel ns, 10 reps), by the
# tile-rows per quantum it yields:
#   [1,1,16384,64] (block_width 2)  1: 26651  2: 26468  4: 25590  8 (uncapped): 28724
#   [1,1,16384,32] (block_width 1)  1: 19155  2: 19656  4: 18212  8 (uncapped): 20531
# Wider tile-rows (block_width >= 8, e.g. [1,1,8192,256]) measured flat-to-worse when
# coarsened, hence the floor is in tiles, not in tile-rows.
QUANTUM_MIN_TILES = 8

# Per-core CB budget in bytes; the only thing low_l1 changes (l1_ledger.md -> Footprint).
CB_BUDGET_BYTES = {False: 524288, True: 65536}

# Fast-tilize eligibility cap on block_width (tilize_helpers.inl: block_width_tiles < 256).
FAST_TILIZE_MAX_BLOCK_WIDTH = 255

# Number of column groups the tile_col axis is split into across Tensix cores.
# Phase 0 = 1 (row split); the grid_2d_split refinement replaces this with the
# assignment rule pinned in op_design.md.
NUM_COL_GROUPS = 1

# Split-reader knob (op_design.md perf lamp "Reader issue-rate"). When a core's
# stick-segment reads are small, one RISC-V is issue-bound; BRISC (the writer,
# which issues only valid_width tile writes per tile-row) then also produces the
# odd tile-rows through its own CB. Engaged when the stick segment is at most
# this many bytes and the core has at least two tile-rows to alternate.
# 0 disables the split reader entirely.
# Measured on WH 64 cores: correct but slower (19.2 -> 20.2 us on [1,1,16384,32],
# 26.9 -> 31.9 us on [1,1,16384,64]): BRISC then carries the odd reads AND the
# writes, and the writes alone are DRAM-throughput-bound (~130 GB/s aggregate).
# Parked disabled; what still has to shorten first is the write path.
SPLIT_READER_MAX_SEGMENT_BYTES = 0


def _div_up(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _div_up(a, b) * b


def _tile_grid(shape, tile_h):
    """(R, C) of the output tile grid; leading dims fold into R (per-image ceil)."""
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    R = leading * _div_up(int(shape[-2]), tile_h)
    C = _div_up(int(shape[-1]), TILE_WIDTH)
    return R, C


def _col_align_tiles(input_tensor, in_elem_bytes):
    """Tile-columns per NoC-alignment unit for stick-segment offsets (source and L1 destination)."""
    src_align = (
        ttnn.get_dram_alignment()
        if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        else ttnn.get_l1_alignment()
    )
    align_bytes = max(src_align, ttnn.get_l1_alignment())
    return max(1, _div_up(align_bytes, TILE_WIDTH * in_elem_bytes))


def balanced_width(core_col_tiles_max, *, per_col_tile_bytes, low_l1, col_align_tiles):
    """op_design.md `balanced_width`: coarsest column block that fits the CB budget, balanced over blocks."""
    cap = min(FAST_TILIZE_MAX_BLOCK_WIDTH, CB_BUDGET_BYTES[low_l1] // per_col_tile_bytes)
    cap = max(col_align_tiles, (cap // col_align_tiles) * col_align_tiles)
    num_col_blocks = _div_up(core_col_tiles_max, cap)
    width = _round_up(_div_up(core_col_tiles_max, num_col_blocks), col_align_tiles)
    return min(width, cap)


def _column_groups(C, num_col_groups, col_align_tiles):
    """[(col_start, core_col_tiles)] balanced over `num_col_groups`, in col_align_tiles units."""
    col_units = _div_up(C, col_align_tiles)
    groups = []
    start_unit = 0
    base, rem = divmod(col_units, num_col_groups)
    for g in range(num_col_groups):
        units = base + (1 if g < rem else 0)
        col_start = start_unit * col_align_tiles
        col_end = min(C, (start_unit + units) * col_align_tiles)
        groups.append((col_start, col_end - col_start))
        start_unit += units
    return groups


_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _shard_rects(tensor, rows_total, width):
    """Per-core valid shard rectangles of an L1-sharded tensor over its 2-D fold.

    Returns ((shard_h, shard_w), [(core, r0, nr, c0, nc), ...]) in elements
    (rows = folded sticks, cols = elements of a stick), cores with no data
    omitted; or None when the shard has no one-rectangle-per-core 2-D
    description (interleaved, DRAM, an ND spec without a 2-D equivalent, or
    more shards than cores). Shard -> core follows the shard orientation:
    HEIGHT / WIDTH enumerate the grid row-wise (ROW_MAJOR) or column-wise
    (COL_MAJOR); BLOCK maps shard (i, j) to core (x0 + j, y0 + i) for ROW_MAJOR
    and (x0 + i, y0 + j) for COL_MAJOR.
    """
    mc = tensor.memory_config()
    spec = mc.shard_spec
    if mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout not in _SHARDED_LAYOUTS or spec is None:
        return None
    shard_h, shard_w = (int(d) for d in spec.shape)
    row_major = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    n_h, n_w = _div_up(rows_total, shard_h), _div_up(width, shard_w)
    if mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        start = spec.grid.bounding_box().start
        placed = [
            (ttnn.CoreCoord(start.x + j, start.y + i) if row_major else ttnn.CoreCoord(start.x + i, start.y + j), i, j)
            for i in range(n_h)
            for j in range(n_w)
        ]
    else:
        cores = ttnn.corerange_to_cores(spec.grid, None, row_major)
        shards = [(k, 0) for k in range(n_h)] if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else None
        if shards is None:
            shards = [(0, k) for k in range(n_w)]
        if len(shards) > len(cores) or n_h * n_w != len(shards):
            return None
        placed = [(cores[k], i, j) for k, (i, j) in enumerate(shards)]
    rects = []
    for core, i, j in placed:
        nr = min(shard_h, rows_total - i * shard_h)
        nc = min(shard_w, width - j * shard_w)
        if nr > 0 and nc > 0:
            rects.append((core, i * shard_h, nr, j * shard_w, nc))
    return (shard_h, shard_w), rects


def _resident_ok(shard_rects, tile_h):
    """op_design.md `resident_ok`: a one-rectangle-per-core 2-D shard, tile-aligned on both axes."""
    if shard_rects is None:
        return False
    (shard_h, shard_w), rects = shard_rects
    return shard_h % tile_h == 0 and shard_w % TILE_WIDTH == 0 and len(rects) > 0


def _rects_key(shard_rects):
    """Hashable identity of a shard placement: nominal shard shape + every core's valid rectangle."""
    shard_shape, rects = shard_rects
    return shard_shape, [((core.x, core.y), r0, nr, c0, nc) for core, r0, nr, c0, nc in rects]


def _core_assignment(input_tensor, output_tensor, *, rows_total, width, tile_h, max_block_width):
    """(assignment, input_resident, output_resident, shard_block_width).

    `assignment` is [(core, row_start, core_row_tiles, col_start, core_col_tiles)]
    over the output tile grid, or None for the interleaved row split. A shard
    wider than `max_block_width` tiles cannot be one resident block (the
    streamed partner CB would not fit the budget), so it is streamed instead.
    """

    def usable(tensor):
        rects = _shard_rects(tensor, rows_total, width)
        if not _resident_ok(rects, tile_h) or rects[0][1] // TILE_WIDTH > max_block_width:
            return None
        return rects

    out_rects, in_rects = usable(output_tensor), usable(input_tensor)
    owner = out_rects if out_rects is not None else in_rects
    if owner is None:
        return None, False, False, None
    (_, shard_w), rects = owner
    assignment = [
        (core, r0 // tile_h, _div_up(nr, tile_h), c0 // TILE_WIDTH, _div_up(nc, TILE_WIDTH))
        for core, r0, nr, c0, nc in rects
    ]
    output_resident = out_rects is not None
    input_resident = in_rects is not None and (not output_resident or _rects_key(in_rects) == _rects_key(out_rects))
    return assignment, input_resident, output_resident, shard_w // TILE_WIDTH


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    tile_h: int = 32,
    in_tile_h: int | None = None,
    low_l1: bool = False,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()

    # ---------------- geometry ----------------
    shape = list(input_tensor.shape)
    R, C = _tile_grid(shape, tile_h)
    rows_total = R * tile_h  # folded sticks (tile-aligned H: every image's H is a multiple of tile_h)
    width = int(shape[-1])

    in_elem_bytes = input_tensor.element_size()
    in_tile_bytes = tile_h * TILE_WIDTH * in_elem_bytes  # tile-sized page holding tile_h stick segments
    out_tile_bytes = output_tensor.buffer_page_size()  # one TILE page of the output dtype
    stick_page_bytes = input_tensor.buffer_aligned_page_size()  # input page stride (a stick, or a shard-width stick)
    # A WIDTH / BLOCK / ND-sharded Layout::ROW_MAJOR input cuts every stick into pages of the
    # shard width; a stick-segment read then splits at page boundaries (reader CT args).
    in_page_bytes = input_tensor.buffer_page_size()
    pages_per_stick = _div_up(width * in_elem_bytes, in_page_bytes)

    # ---------------- block knobs ----------------
    col_align_tiles = _col_align_tiles(input_tensor, in_elem_bytes)

    def _per_col_tile_bytes(num_input_cbs, *, input_resident=False, output_resident=False):
        """Bytes of STREAMED CBs per tile-column of one tile-row (resident CBs are the tensor's own L1)."""
        streamed_in = 0 if input_resident else num_input_cbs * DEPTH_IN * in_tile_bytes
        streamed_out = 0 if output_resident else DEPTH_OUT * out_tile_bytes
        return streamed_in + streamed_out

    # ---------------- core assignment (op_design.md -> Regimes, selection function) ----------------
    # A resident shard is ONE block (block_width = shard width in tiles), so its streamed partner
    # CB must hold depth * shard-width tiles within the budget.
    max_shard_block_width = min(FAST_TILIZE_MAX_BLOCK_WIDTH, CB_BUDGET_BYTES[low_l1] // _per_col_tile_bytes(1))
    assignment, input_resident, output_resident, shard_block_width = _core_assignment(
        input_tensor,
        output_tensor,
        rows_total=rows_total,
        width=width,
        tile_h=tile_h,
        max_block_width=max_shard_block_width,
    )
    any_resident = input_resident or output_resident

    if assignment is None:
        # row_split_interleaved (also every streamed-only sharded case: DRAM-sharded sides,
        # ND specs without a 2-D equivalent). The Tensix core count is read from the device.
        assert NUM_COL_GROUPS == 1, "grid_2d_split (NUM_COL_GROUPS > 1) is a deferred regime"
        col_groups = _column_groups(C, NUM_COL_GROUPS, col_align_tiles)
        col_start, core_col_tiles = col_groups[0]  # NUM_COL_GROUPS == 1: every core owns [0, C)
        grid = device.compute_with_storage_grid_size()
        (_n, all_cores, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, R, row_wise=True)
        assignment = []
        row_start = 0
        for group, rows in ((core_group_1, rows_g1), (core_group_2, rows_g2)):
            for core in ttnn.corerange_to_cores(group, None, True):
                assignment.append((core, row_start, rows, col_start, core_col_tiles))
                row_start += rows
        assert row_start == R, f"row split covered {row_start} of {R} tile-rows"
    else:
        # sharded_resident: the owning shard grid (only the Tensix cores holding data).
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, *_ in assignment])

    core_row_tiles_max = max(rows for _, _, rows, _, _ in assignment)
    core_col_tiles_max = max(cols for _, _, _, _, cols in assignment)

    def _block_width_for(num_input_cbs):
        if any_resident:
            return shard_block_width  # the whole resident shard width is one block
        return balanced_width(
            core_col_tiles_max,
            per_col_tile_bytes=_per_col_tile_bytes(num_input_cbs),
            low_l1=low_l1,
            col_align_tiles=col_align_tiles,
        )

    # The split reader decision needs the segment width, which needs block_width, which
    # needs the CB count: resolve it for the split configuration first (the tighter
    # budget), and keep the split only if its segments are small enough. It only
    # applies when both sides stream.
    block_width_split = _block_width_for(2)
    split_segment_bytes = block_width_split * TILE_WIDTH * in_elem_bytes
    num_col_blocks_max = _div_up(core_col_tiles_max, block_width_split)
    split_reader = (
        not any_resident
        and split_segment_bytes <= SPLIT_READER_MAX_SEGMENT_BYTES
        and core_row_tiles_max * num_col_blocks_max >= 2
    )
    num_input_cbs = 2 if split_reader else 1
    block_width = block_width_split if split_reader else _block_width_for(1)
    assert core_col_tiles_max <= block_width or not any_resident

    # Streaming window along tile_row (the streamed side only): fill what the budget leaves
    # after block_width, but keep at least DEPTH_IN quanta per core so the double buffer still
    # overlaps (one quantum per core serializes read -> tilize -> write). The split reader
    # alternates CBs per tile-row, so it keeps one tile-row per quantum.
    per_row_bytes = block_width * _per_col_tile_bytes(
        num_input_cbs, input_resident=input_resident, output_resident=output_resident
    )
    max_positions = core_row_tiles_max * _div_up(core_col_tiles_max, block_width)  # busiest core's walk length
    # QUANTUM_MIN_TILES counts full 32-row tiles: a tiny tile carries tile_h / 32 of one
    # tile's bytes and per-quantum costs are per handshake, so the floor scales with 32 / tile_h
    # (at tile_h = 1 one "tile" is a single stick segment).
    quantum_min_tiles = QUANTUM_MIN_TILES * (FULL_TILE_HEIGHT // tile_h)
    if split_reader:
        rows_per_quantum = 1
    else:
        rows_per_quantum = max(
            1,
            min(
                _div_up(quantum_min_tiles, block_width),
                max_positions // DEPTH_IN,
                CB_BUDGET_BYTES[low_l1] // max(1, per_row_bytes),
            ),
        )

    # ---------------- circular buffers ----------------
    tile_desc = ttnn.TileDescriptor(tile_h, TILE_WIDTH)
    quantum_tiles = rows_per_quantum * block_width  # pages per CB push / pop on a streamed CB
    input_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_INPUT_STICKS,
        data_format=input_tensor.dtype,
        page_size=in_tile_bytes,
        tile=tile_desc,
    )
    if input_resident:
        # Zero-copy: cb_input_sticks IS the input shard. A Layout::ROW_MAJOR shard row has
        # stride shard_w * elem_bytes = block_width * tile_col_bytes, exactly the tilize input
        # layout, so tile_h sticks of the shard are block_width tile-sized pages.
        shard_h = input_tensor.memory_config().shard_spec.shape[0]
        cb_input_sticks = ttnn.cb_descriptor_from_sharded_tensor(
            CB_INPUT_STICKS,
            input_tensor,
            total_size=(shard_h // tile_h) * block_width * in_tile_bytes,
            core_ranges=all_cores,
        )
        cb_input_sticks.format_descriptors = [input_format]
    else:
        cb_input_sticks = ttnn.CBDescriptor(
            total_size=DEPTH_IN * quantum_tiles * in_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[input_format],
        )
    cbs = [cb_input_sticks]
    if split_reader:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=DEPTH_IN * quantum_tiles * in_tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_STICKS_ODD,
                        data_format=input_tensor.dtype,
                        page_size=in_tile_bytes,
                        tile=tile_desc,
                    )
                ],
            )
        )
    assert len(cbs) == num_input_cbs
    if output_resident:
        # Zero-copy: compute packs straight into the output shard (TILE pages, shard order).
        cb_output_tiles = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor, core_ranges=all_cores)
    else:
        cb_output_tiles = ttnn.CBDescriptor(
            total_size=DEPTH_OUT * quantum_tiles * out_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=out_tile_bytes,
                    tile=tile_desc,
                )
            ],
        )

    # ---------------- kernel args ----------------
    # CT args: config only (dtype pair, tile, block_width, residency, accessor args) ->
    # program-cache friendly; buffer addresses ride on RT args and on the resident CBs.
    tile_col_bytes = TILE_WIDTH * in_elem_bytes
    assert 1 <= READ_AHEAD <= DEPTH_IN
    reader_ct_args = [
        CB_INPUT_STICKS,
        block_width,
        tile_h,
        tile_col_bytes,
        stick_page_bytes,
        int(split_reader),
        DEPTH_IN,
        READ_AHEAD,
        rows_per_quantum,
        int(input_resident),
        in_page_bytes,
        pages_per_stick,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [
        CB_OUTPUT_TILES,
        block_width,
        out_tile_bytes,
        int(split_reader),
        CB_INPUT_STICKS_ODD,
        tile_h,
        tile_col_bytes,
        stick_page_bytes,
        DEPTH_IN,
        READ_AHEAD,
        rows_per_quantum,
        int(output_resident),
        in_page_bytes,
        pages_per_stick,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    compute_ct_args = [CB_INPUT_STICKS, CB_OUTPUT_TILES, block_width, int(split_reader), CB_INPUT_STICKS_ODD]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    for core_idx, (core, row_start, core_row_tiles, col_start, core_col_tiles) in enumerate(assignment):
        # Per-core traversal rotation (single source for reader AND writer): spreads the
        # cores' concurrent stick reads / tile writes over the DRAM banks. A resident side
        # fixes the tile-row order to shard order, so only the in-tile-row stick order rotates.
        row_rotation = 0 if any_resident else core_idx
        stick_rotation = core_idx
        reader_rt_args[core.x][core.y] = [
            in_addr,
            row_start,
            core_row_tiles,
            col_start,
            core_col_tiles,
            row_rotation,
            stick_rotation,
        ]
        writer_rt_args[core.x][core.y] = [
            out_addr,
            row_start,
            core_row_tiles,
            col_start,
            core_col_tiles,
            C,
            row_rotation,
            in_addr,
            stick_rotation,
        ]
        compute_rt_args[core.x][core.y] = [core_row_tiles, core_col_tiles]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),  # BRISC / NoC1
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        # 16-bit DEST matches Float16_b pages; half-sync DEST is a fast-tilize requirement.
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=False),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs + [cb_output_tiles],
    )
