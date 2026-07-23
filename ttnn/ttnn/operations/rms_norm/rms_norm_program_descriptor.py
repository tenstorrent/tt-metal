# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for rms_norm.

Row-parallel, bounded two-pass streaming reduce over W (op_design.md §1, §5-§9):

  Pass 1: x -> square -> accumulate_reduce(SUM, 1/W)  -> cb_rstd = mean(x^2)
          cb_rstd -> (+eps, rsqrt)                    -> cb_rstd = 1/RMS  (held)
  Pass 2: x -> mul<Col>(x, rstd)                       -> cb_norm
          cb_norm -> mul<Row>(norm, gamma) / copy      -> cb_out

Work distribution: the R = NC*ceil(H/32) independent tile-rows are split across
the whole compute grid via `split_work_to_cores(R, grid, row_wise=True)`; each
core owns a contiguous run [row_start, row_start+num_rows) and loops its rows,
each row a 2-pass streaming reduce over NUM_BLOCKS blocks of BLOCK_SIZE tiles.

Every block knob is a live parameter (single source of truth):
  BLOCK_SIZE = pick_block_size(Wt)   -> CT arg (reader, compute)
  NUM_BLOCKS = Wt // BLOCK_SIZE       -> derived
  DEPTH      = 2                      -> CB depth (num_pages = DEPTH*BLOCK_SIZE)
No CB is sized by an op dimension (Wt/W/H/R); cb_rstd is 1 tile per row.

RM regime uses the tilize/untilize dataflow helpers
(dataflow_kernel_lib::read_sticks_for_tilize / write_sticks_after_untilize),
which handle non-tile-aligned W (row_bytes) and H (partial last block) and the
per-core start-row offset natively — no host-side pad/slice.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32
DEPTH = 2  # per-streaming-CB double-buffer depth (op_design.md §1)

# ---- Resident single-read fast-path knobs (op_design.md §1 lamp 1; Refinement 3) ----
# RESIDENT_X_DEPTH is the MAX resident input-CB depth (whole tile-rows) the host will
# use: at depth d the reader prefetches d-1 tile-rows ahead while compute reads the
# current one from L1. The host picks the largest depth in [1, RESIDENT_X_DEPTH] that
# fits L1_RESIDENT_BUDGET, so wide rows single-buffer and narrow rows double-buffer.
# L1_RESIDENT_BUDGET is the per-core CB budget the resident footprint must fit under —
# the explicit predicate that keeps the resident path from overflowing L1 (Blackhole
# L1 ~1.5 MB; a conservative 1.1 MB CB budget). Because the intermediates stay
# per-block, only cb_x_in + cb_gamma scale with Wt, so every prefill width fits. Both
# are live tunables with a single source of truth — no CB literal derives independently.
RESIDENT_X_DEPTH = 2
L1_RESIDENT_BUDGET = 1_100_000


def _pick_block_size(Wt: int) -> int:
    """Largest divisor of Wt that is <= 8 (the double_buffer sweet spot; not 1).

    Phase-1 value of the BLOCK_SIZE knob (op_design.md §1). Kept a function so a
    later refinement can raise the cap / change the policy in one place.
    """
    for candidate in range(min(8, Wt), 0, -1):
        if Wt % candidate == 0:
            return candidate
    return 1


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _elem_size(tensor: ttnn.Tensor) -> int:
    """Bytes per element — used ONLY for RM stick-byte math (`cols * elem`).

    Block-float dtypes (bfloat8_b) have no fixed per-element size (16 values
    share one exponent), so `element_size()` raises for them. bf8b is TILE-only
    (bf8b+RM is INVALID), so the RM regime — the only consumer of this value —
    never runs for it; return 0 as an unused placeholder instead of raising.
    Page-size math (below) uses `buffer_aligned_page_size()`, which is correct
    for block-float (returns the tile page), so it is left as-is.
    """
    try:
        return tensor.element_size()
    except (ValueError, RuntimeError):
        return 0


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: "ttnn.Tensor | None" = None,
    epsilon: float = 1e-6,
    compute_kernel_config: "ttnn.ComputeConfigDescriptor | None" = None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()

    # Cross-core W-split scheme (op_design.md §1 lamp 2, §5): WIDTH/BLOCK-sharded
    # inputs pre-place the hidden dim across cores, so the RMS reduce is cross-core.
    # Dispatched to a separate builder + kernel set; the interleaved row-parallel
    # path below is untouched (Refinement 4).
    _ml = input_tensor.memory_config().memory_layout
    if _ml in (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED):
        return _create_sharded_xcore_descriptor(
            input_tensor,
            output_tensor,
            gamma=gamma,
            epsilon=epsilon,
            compute_kernel_config=compute_kernel_config,
        )

    shape = list(input_tensor.shape)

    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    # Alignment-aware tile geometry (per-image ceil; op_design.md §6).
    Ht_img = (origin_H + TILE_DIM - 1) // TILE_DIM
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    R = NC * Ht_img
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    BLOCK_SIZE = _pick_block_size(Wt)
    NUM_BLOCKS = Wt // BLOCK_SIZE

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    # gamma layout is an independent knob from the input layout (a valid TARGET
    # cell is RM input + TILE gamma). RM gamma -> reader reads sticks + compute
    # tilizes; TILE gamma -> reader reads tiles straight into cb_gamma and the
    # compute-side gamma tilize is skipped (op_design.md §5 knob-turn).
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    inv_N_bits = _f32_bits(1.0 / float(origin_W))  # scaler = 1/origin_W (true element count)
    eps_bits = _f32_bits(epsilon)

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    in_elem = _elem_size(input_tensor)  # RM-only; 0 placeholder for block-float
    out_elem = _elem_size(output_tensor)
    in_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_elem = _elem_size(gamma)  # RM-only; 0 placeholder for block-float
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
    else:
        gamma_dtype = in_dtype
        gamma_elem = in_elem
        gamma_page = in_page
        tile_gamma = tile_in

    # ---- Resident single-read fast-path predicate (op_design.md §1 lamp 1) ----
    # When the tile-row's x (Wt tiles) + resident gamma fit one core's L1, load x
    # ONCE and read it from L1 in BOTH passes — eliminating the 2nd DRAM read of x —
    # and hold gamma resident (read once per core instead of re-read per tile-row).
    # The compute reads each block at the absolute front offset b*BLOCK_SIZE, so the
    # intermediates (cb_xsq / cb_norm) stay 2*BLOCK_SIZE (NOT sized by Wt) and every
    # prefill width fits. Only cb_x_in and cb_gamma are sized by Wt — the design's
    # sanctioned dual-path exception, gated by the explicit L1 budget. The streaming
    # two-pass path (below) is the fallback. TILE input only (+ TILE / no gamma —
    # RM gamma keeps its tilize path); the perf config is TILE input + TILE gamma.
    is_tile_input = (not is_rm) and (not gamma_is_rm)

    def _resident_l1(x_depth):
        # cb_x_in (x_depth whole rows) + cb_gamma (1 row) + small streaming CBs.
        return (
            x_depth * Wt * tile_in
            + (Wt * tile_gamma if has_gamma else 0)
            + 2 * BLOCK_SIZE * tile_in  # cb_xsq  (per-block)
            + 2 * BLOCK_SIZE * tile_in  # cb_norm (per-block)
            + DEPTH * BLOCK_SIZE * tile_out  # cb_out  (per-block, streamed)
            + max(DEPTH, 2) * tile_fp32  # cb_rstd
            + (2 if has_partial_w else 1) * tile_bf16  # cb_scaler
        )

    # Pick the largest input-CB depth that fits the budget (RESIDENT_X_DEPTH..1) so
    # the reader prefetches the next tile-row where L1 allows, and single-buffers the
    # widest rows that only fit at depth 1. Single source of truth for the depth.
    resident_x_depth = 0
    for d in range(RESIDENT_X_DEPTH, 0, -1):
        if _resident_l1(d) <= L1_RESIDENT_BUDGET:
            resident_x_depth = d
            break
    use_resident = is_tile_input and (resident_x_depth > 0)

    # CB page counts branch on the path (single source of truth for each CB). Only
    # cb_x_in and cb_gamma are resident-sized; the intermediates stay per-block.
    x_in_pages = resident_x_depth * Wt if use_resident else DEPTH * BLOCK_SIZE
    xsq_pages = 2 * BLOCK_SIZE
    norm_pages = 2 * BLOCK_SIZE
    gamma_cb_pages = Wt if use_resident else DEPTH * BLOCK_SIZE
    out_pages = DEPTH * BLOCK_SIZE

    # ---- Work distribution: split R tile-rows across the whole grid ----
    grid_size = device.compute_with_storage_grid_size()
    (
        _num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_g1,
        rows_per_core_g2,
    ) = ttnn.split_work_to_cores(grid_size, R, row_wise=True)

    assignment = []  # (core, row_start, num_rows)
    start = 0
    for group, per_core in ((core_group_1, rows_per_core_g1), (core_group_2, rows_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core

    # ---- Circular buffers (op_design.md §7); no CB sized by an op dimension ----
    CB_X_STICKS = 0
    CB_X_IN = 1
    CB_SCALER = 2
    CB_GAMMA = 3
    CB_GAMMA_STICKS = 4
    CB_OUT = 16
    CB_OUT_STICKS = 17
    CB_XSQ = 24
    CB_RSTD = 25
    CB_NORM = 26

    cbs = []

    def add_cb(idx, page_size, num_pages, fmt):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    # input tiles. streaming: DEPTH*BLOCK_SIZE (both passes re-read from DRAM).
    # resident: resident_x_depth*Wt — whole tile-row(s) held so pass 2 reads L1, not DRAM.
    add_cb(CB_X_IN, tile_in, x_in_pages, in_dtype)
    # reduce scaler: 1/W (+ partial tile), bf16, wait-not-pop across all rows
    add_cb(CB_SCALER, tile_bf16, 2 if has_partial_w else 1, ttnn.bfloat16)
    # pass-1 intermediate x^2 (square -> reduce), per-block (2*BLOCK_SIZE) both paths
    add_cb(CB_XSQ, tile_in, xsq_pages, in_dtype)
    # 1/RMS (1 tile/row), fp32 accumulate, held across pass 2
    add_cb(CB_RSTD, tile_fp32, max(DEPTH, 2), ttnn.float32)
    # pass-2 intermediate x*rstd (mul<Col> -> mul<Row>), per-block (2*BLOCK_SIZE) both paths
    add_cb(CB_NORM, tile_in, norm_pages, in_dtype)
    # output tiles (TILE: -> writer; RM: -> untilize), per-block streamed (DEPTH*BLOCK_SIZE)
    add_cb(CB_OUT, tile_out, out_pages, out_dtype)

    if is_rm:
        # RM x: raw sticks packed for compute-side tilize (tile-paged, TILE granularity)
        add_cb(CB_X_STICKS, tile_in, DEPTH * BLOCK_SIZE, in_dtype)
        # RM out: untilized row-major (tile-paged output of compute untilize)
        add_cb(CB_OUT_STICKS, tile_out, DEPTH * BLOCK_SIZE, out_dtype)

    if has_gamma:
        # cb_gamma holds tiles in both regimes: TILE gamma -> reader is the
        # producer (direct tile read); RM gamma -> compute-tilize is the producer.
        # Single producer per compiled program (CT-arg dispatch), same pattern as
        # cb_x_in for the input layout.
        add_cb(CB_GAMMA, tile_gamma, gamma_cb_pages, gamma_dtype)
        # cb_gamma_sticks is the RM-gamma-only tilize input; not needed for TILE gamma.
        if gamma_is_rm:
            add_cb(CB_GAMMA_STICKS, tile_gamma, DEPTH * BLOCK_SIZE, gamma_dtype)

    # ---- Reader kernel ----
    reader_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        1 if is_rm else 0,
        1 if has_gamma else 0,
        in_elem,
        gamma_elem,
        in_page,
        gamma_page,
        1 if gamma_is_rm else 0,
        1 if use_resident else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    for core, row_start, num_rows in assignment:
        reader_rt_args[core.x][core.y] = [in_addr, gamma_addr, row_start, num_rows]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel ----
    writer_ct_args = [
        Ht_img,
        Wt,
        BLOCK_SIZE,
        NUM_BLOCKS,
        origin_H,
        origin_W,
        1 if is_rm else 0,
        out_elem,
        out_page,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    out_addr = output_tensor.buffer_address()
    for core, row_start, num_rows in assignment:
        writer_rt_args[core.x][core.y] = [out_addr, row_start, num_rows]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel ----
    compute_ct_args = [
        BLOCK_SIZE,
        NUM_BLOCKS,
        1 if is_rm else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        eps_bits,
        1 if gamma_is_rm else 0,
        1 if use_resident else 0,
    ]

    compute_rt_args = ttnn.RuntimeArgs()
    for core, row_start, num_rows in assignment:
        compute_rt_args[core.x][core.y] = [num_rows]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )


# ---------------------------------------------------------------------------
# Cross-core W-split builder (op_design.md §1 lamp 2, §5) — WIDTH/BLOCK sharded
# ---------------------------------------------------------------------------
# The hidden dim W is pre-placed across a group of K cores. Each core reduces its
# local W-slice to a partial Σx²·(1/W); one cross-core round per tile-row (gather
# to the group master -> master folds K partials + finalize (+eps, rsqrt) -> the
# 1/RMS is broadcast back) precedes the per-core normalize. Slices are consumed
# LOCALLY via zero-copy sharded CBs (never re-read through a TensorAccessor). The
# reduction group is:
#   WIDTH_SHARDED : ALL cores (every core owns a W-slice of the SAME full-height
#                   rows) -> one group, master = shard core 0.
#   BLOCK_SHARDED : the cores of one grid ROW (same tile-rows, disjoint W-slices)
#                   -> one group per grid row, master = the row's x=0 core.
# All-unicast transport (topology-agnostic; NoC-mcast/two-stage is the R6 lever).

# Semaphore ids (monotone counters; reused across disjoint BLOCK groups — a sem id
# resolves to a per-(core,id) L1 cell, so disjoint groups never interfere, §5).
_SEM_GATHER = 0
_SEM_BCAST = 1
_SEM_DONE = 2


def _create_sharded_xcore_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma=None,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    device = input_tensor.device()
    shape = list(input_tensor.shape)
    origin_W = int(shape[-1])
    origin_H = int(shape[-2])
    NC = 1
    for d in shape[:-2]:
        NC *= int(d)

    Ht_img = (origin_H + TILE_DIM - 1) // TILE_DIM
    Wt = (origin_W + TILE_DIM - 1) // TILE_DIM
    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0

    has_gamma = gamma is not None
    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype

    mem = input_tensor.memory_config()
    ml = mem.memory_layout
    ss = mem.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    per_h_t = sh // TILE_DIM  # tile-rows this core holds
    per_w_t = sw // TILE_DIM  # W-tiles this core holds
    grid = ss.grid
    cores = ttnn.corerange_to_cores(grid, None, True)  # row-major shard order

    def _v(core):
        vc = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
        return int(vc.x), int(vc.y)

    # ---- Per-core group assignment (single source of truth for the topology) ----
    # entry: (core, slice_index, master_core, is_partial_holder, w_tile_start, vwt)
    Ht_local = per_h_t
    entries = []
    if ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        # One group; every core owns a W-slice of the full-height rows.
        K = len(cores)
        master = cores[0]
        for i, c in enumerate(cores):
            w_tile_start = i * per_w_t
            vwt = max(0, min(per_w_t, Wt - w_tile_start))
            entries.append((c, i, master, i == len(cores) - 1, w_tile_start, vwt))
    else:  # BLOCK_SHARDED — one group per grid row
        bb = grid.bounding_box()
        x0, y0 = int(bb.start.x), int(bb.start.y)
        x1 = int(bb.end.x)
        nx = x1 - x0 + 1
        K = nx
        for c in cores:
            xrel = int(c.x) - x0
            master = ttnn.CoreCoord(x0, int(c.y))
            w_tile_start = xrel * per_w_t
            vwt = max(0, min(per_w_t, Wt - w_tile_start))
            entries.append((c, xrel, master, xrel == nx - 1, w_tile_start, vwt))

    # workers per master (for the master's broadcast list), keyed by (mx, my)
    workers_of = {}
    for c, slice_index, master, _iph, _wts, _vwt in entries:
        key = (int(master.x), int(master.y))
        if slice_index != 0:
            workers_of.setdefault(key, []).append(c)

    inv_N_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    tile_in = ttnn.tile_size(in_dtype)
    tile_out = ttnn.tile_size(out_dtype)
    tile_bf16 = ttnn.tile_size(ttnn.bfloat16)
    tile_fp32 = ttnn.tile_size(ttnn.float32)

    if has_gamma:
        gamma_dtype = gamma.dtype
        gamma_page = gamma.buffer_aligned_page_size()
        tile_gamma = ttnn.tile_size(gamma_dtype)
    else:
        gamma_dtype = in_dtype
        gamma_page = input_tensor.buffer_aligned_page_size()
        tile_gamma = tile_in

    all_cores = grid

    # ---- Circular buffers (all bounded; no CB grows with the tile-row count) ----
    CB_X_IN = 1
    CB_SCALER = 2
    CB_GAMMA = 3
    CB_GATHER = 5
    CB_STAT_HANDOFF = 6
    CB_STAT_GLOBAL = 7
    CB_OUT = 16
    CB_XSQ = 24
    CB_STAT_LOCAL = 25
    CB_NORM = 26

    cbs = [
        # zero-copy sharded input W-slice + output W-slice (consumed/produced locally)
        ttnn.cb_descriptor_from_sharded_tensor(CB_X_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]

    def add_cb(idx, page_size, num_pages, fmt):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    add_cb(CB_SCALER, tile_bf16, 2 if has_partial_w else 1, ttnn.bfloat16)
    # pass-1 squares the whole vwt-tile block before the single block-reduce, so
    # cb_xsq must hold a full W-slice block (2*per_w_t double-buffers it).
    add_cb(CB_XSQ, tile_in, 2 * per_w_t, in_dtype)
    add_cb(CB_STAT_LOCAL, tile_fp32, 2, ttnn.float32)
    add_cb(CB_GATHER, tile_fp32, K, ttnn.float32)  # fixed-base fan-in (K partials/round)
    add_cb(CB_STAT_HANDOFF, tile_fp32, 2, ttnn.float32)
    add_cb(CB_STAT_GLOBAL, tile_fp32, 1, ttnn.float32)  # depth 1 -> fixed base for the mcast-back
    add_cb(CB_NORM, tile_in, 2, in_dtype)
    if has_gamma:
        add_cb(CB_GAMMA, tile_gamma, per_w_t, gamma_dtype)

    # ---- Reader (scaler + gamma W-slice) ----
    reader_ct = [
        1 if has_gamma else 0,
        inv_N_bits,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        gamma_page,
    ]
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_rt = ttnn.RuntimeArgs()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    for c, _si, _m, _iph, w_tile_start, vwt in entries:
        reader_rt[c.x][c.y] = [gamma_addr, w_tile_start, vwt]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_xcore_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer (cross-core transport) ----
    writer_ct = [_SEM_GATHER, _SEM_BCAST, _SEM_DONE, Ht_local, K, tile_fp32]
    writer_rt = ttnn.RuntimeArgs()
    for c, slice_index, master, _iph, _wts, _vwt in entries:
        mvx, mvy = _v(master)
        is_master = 1 if slice_index == 0 else 0
        row = [is_master, slice_index, mvx, mvy]
        if is_master:
            wl = workers_of.get((int(master.x), int(master.y)), [])
            row.append(len(wl))
            for w in wl:
                wvx, wvy = _v(w)
                row.extend([wvx, wvy])
        else:
            row.append(0)
        writer_rt[c.x][c.y] = row
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_xcore_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute (local reduce + master combine + normalize) ----
    compute_ct = [per_w_t, Ht_local, K, 1 if has_gamma else 0, 1 if has_partial_w else 0, eps_bits]
    compute_rt = ttnn.RuntimeArgs()
    for c, slice_index, _m, is_partial_holder, _wts, vwt in entries:
        compute_rt[c.x][c.y] = [vwt, 1 if is_partial_holder else 0, 1 if slice_index == 0 else 0]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_xcore_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor(),
    )

    semaphores = [
        ttnn.SemaphoreDescriptor(id=_SEM_GATHER, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_BCAST, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_DONE, core_ranges=all_cores, initial_value=0),
    ]

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )
