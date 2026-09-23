# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor for the `row_split_interleaved` regime (op_design.md).

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


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    tile_h: int = 32,
    low_l1: bool = False,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()

    # ---------------- geometry ----------------
    shape = list(input_tensor.shape)
    R, C = _tile_grid(shape, tile_h)

    in_elem_bytes = input_tensor.element_size()
    in_tile_bytes = tile_h * TILE_WIDTH * in_elem_bytes  # tile-sized page holding tile_h stick segments
    out_tile_bytes = output_tensor.buffer_page_size()  # one TILE page of the output dtype
    stick_page_bytes = input_tensor.buffer_aligned_page_size()  # interleaved stick page stride

    # ---------------- block knobs ----------------
    col_align_tiles = _col_align_tiles(input_tensor, in_elem_bytes)
    col_groups = _column_groups(C, NUM_COL_GROUPS, col_align_tiles)
    core_col_tiles_max = max(n for _, n in col_groups)

    # The split reader decision needs the segment width, which needs block_width, which
    # needs the CB count: resolve it for the split configuration first (the tighter
    # budget), and keep the split only if its segments are small enough.
    def _per_col_tile_bytes(num_input_cbs):
        """CB bytes per tile-column of one tile-row, over every CB and its depth."""
        return num_input_cbs * DEPTH_IN * in_tile_bytes + DEPTH_OUT * out_tile_bytes

    def _block_width_for(num_input_cbs):
        return balanced_width(
            core_col_tiles_max,
            per_col_tile_bytes=_per_col_tile_bytes(num_input_cbs),
            low_l1=low_l1,
            col_align_tiles=col_align_tiles,
        )

    block_width_split = _block_width_for(2)
    split_segment_bytes = block_width_split * TILE_WIDTH * in_elem_bytes

    # ---------------- work distribution (tile_row axis) ----------------
    # The Tensix core count is read from the device at runtime, never hardcoded.
    # With NUM_COL_GROUPS == 1 the whole grid splits the tile_row axis; the
    # grid_2d_split refinement assigns (g_r, g_c) groups instead (op_design.md).
    assert NUM_COL_GROUPS == 1, "grid_2d_split (NUM_COL_GROUPS > 1) is a deferred regime"
    grid = device.compute_with_storage_grid_size()
    (
        _num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_g1,
        rows_g2,
    ) = ttnn.split_work_to_cores(grid, R, row_wise=True)

    cores_with_rows = [(c, rows_g1) for c in ttnn.corerange_to_cores(core_group_1, None, True)]
    cores_with_rows += [(c, rows_g2) for c in ttnn.corerange_to_cores(core_group_2, None, True)]

    # Tile-rows each core walks in total (all column blocks): the split alternates on it.
    num_col_blocks_max = _div_up(core_col_tiles_max, block_width_split)
    split_reader = split_segment_bytes <= SPLIT_READER_MAX_SEGMENT_BYTES and rows_g1 * num_col_blocks_max >= 2
    num_input_cbs = 2 if split_reader else 1
    block_width = block_width_split if split_reader else _block_width_for(1)

    # Streaming window along tile_row: fill what the budget leaves after block_width, but
    # keep at least DEPTH_IN quanta per core so the double buffer still overlaps (one
    # quantum per core serializes read -> tilize -> write). The split reader alternates
    # CBs per tile-row, so it keeps one tile-row per quantum.
    per_row_bytes = block_width * _per_col_tile_bytes(num_input_cbs)
    max_positions = rows_g1 * _div_up(core_col_tiles_max, block_width)  # busiest core's walk length
    if split_reader:
        rows_per_quantum = 1
    else:
        rows_per_quantum = max(
            1,
            min(
                _div_up(QUANTUM_MIN_TILES, block_width),
                max_positions // DEPTH_IN,
                CB_BUDGET_BYTES[low_l1] // per_row_bytes,
            ),
        )

    # ---------------- circular buffers ----------------
    tile_desc = ttnn.TileDescriptor(tile_h, TILE_WIDTH)
    quantum_tiles = rows_per_quantum * block_width  # pages per CB push / pop
    cb_input_sticks = ttnn.CBDescriptor(
        total_size=DEPTH_IN * quantum_tiles * in_tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_INPUT_STICKS,
                data_format=input_tensor.dtype,
                page_size=in_tile_bytes,
                tile=tile_desc,
            )
        ],
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
    # CT args: config only (dtype pair, tile, block_width, accessor args) -> program-cache friendly.
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
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    compute_ct_args = [CB_INPUT_STICKS, CB_OUTPUT_TILES, block_width, int(split_reader), CB_INPUT_STICKS_ODD]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()

    col_start, core_col_tiles = col_groups[0]  # NUM_COL_GROUPS == 1: every core owns [0, C)
    row_start = 0
    for core_idx, (core, core_row_tiles) in enumerate(cores_with_rows):
        # Per-core traversal rotation (single source for reader AND writer): spreads the
        # cores' concurrent stick reads / tile writes over the DRAM banks.
        rotation = core_idx
        reader_rt_args[core.x][core.y] = [in_addr, row_start, core_row_tiles, col_start, core_col_tiles, rotation]
        writer_rt_args[core.x][core.y] = [
            out_addr,
            row_start,
            core_row_tiles,
            col_start,
            core_col_tiles,
            C,
            rotation,
            in_addr,
        ]
        compute_rt_args[core.x][core.y] = [core_row_tiles, core_col_tiles]
        row_start += core_row_tiles
    assert row_start == R, f"row split covered {row_start} of {R} tile-rows"

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
