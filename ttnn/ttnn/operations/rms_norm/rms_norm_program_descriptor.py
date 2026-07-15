# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm program descriptor.

Parameterized row-parallel streaming reduction (see kernels/*.cpp). Every block
factor and buffer depth is a knob derived from one source of truth:

  * W_BLOCK_TARGET  — desired W-tiles per reduce chunk (Refinement 3 co-tune).
  * ROW_BLOCK_TILES — tile-rows per outer pass (phase-1: 1).

CB page counts and loop trip counts are computed FROM those knobs and the input
shape — never sized to Wt / W / sequence length, so per-core L1 stays bounded
for arbitrarily wide W.

Refinement 4 (Multi-core row distribution + HEIGHT_SHARDED). The Row axis is
independent, so the tile-row range is distributed across the grid with NO
cross-core communication — a pure per-core runtime-arg change (each core gets its
own (start_tile_row, num_tile_rows); the kernels already iterate `start_tile_row +
t`, no loop-nest change). INTERLEAVED uses `split_work_to_cores`; HEIGHT_SHARDED
uses the same split with the row->core assignment pinned by the shard spec (see
_interleaved_assignment / _height_sharded_assignment).

Refinement 3 (Data-movement co-tune, PERF). The real bottleneck (found by
on-device measurement, not the design's assumption) was NOT NoC bandwidth but
per-tile SYNCHRONIZATION: with W_BLOCK_TILES=1 the reader/writer ping-pong the
input/output CBs one tile at a time (reserve/read/barrier/push per tile), and the
compute runs each helper on a single tile — so the per-tile CB handshake + barrier
+ per-helper init/reconfig overhead dominates. (A DM-payload ablation — stubbing
the NoC reads while KEEPING the per-tile CB signaling — moved device-ns by 0%,
i.e. reads are hidden; the cost is the signaling GRANULARITY, not the transfer.)
The co-tune raises the block to W_BLOCK_TILES tiles, which compounds three levers:
  * compute_block_size — each square/reduce/mul/tilize/untilize helper runs on
    W_BLOCK_TILES tiles per call, amortizing init/reconfig/pipeline fill-drain.
  * double_buffer (reader + writer) — issue a whole block of async reads/writes
    then ONE barrier, and coarsen the reader->compute and compute->writer CB
    handshake W_BLOCK_TILES-fold (this is the dominant win).
  * transfer size (RM regime) — a W_BLOCK_TILES-wide stick slice is one big read
    instead of W_BLOCK_TILES narrow ones.
Measured (median device-ns, W_BLOCK_TARGET 1->8, WH B0, 1 core): TILE bf16
(1,1,32,8192) 283.5us -> 88.1us = 3.22x; RM bf16 same shape 6.43ms -> 0.85ms =
7.5x. Every guard-set path (TILE/RM x gamma/no_gamma x bf16/fp32) improved
2.0x-7.7x, none regressed. reader_placement (row_wise) is deferred to Refinement 4
(needs the multi-core reader). See changelog for the full before/after table.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# --- Blocking-model knobs (single source of truth) ---
# W_BLOCK_TARGET is the DESIRED reduce-chunk width in W-tiles. The effective
# W_BLOCK_TILES is derived per invocation (below) as the largest divisor of Wt
# that is <= this target, so every W-block is uniformly W_BLOCK_TILES tiles (no
# partial last W-block) in BOTH layout regimes and BOTH passes — the templated
# tilize/untilize<W_BLOCK_TILES> and the reader/writer block loops all stay
# uniform, and Wt % W_BLOCK_TILES == 0 holds by construction. Wide-W perf targets
# (Wt=128/256) get the full target; prime/awkward Wt degrade to a smaller divisor
# (small shapes where per-helper overhead is not the bottleneck anyway).
W_BLOCK_TARGET = 8  # phase-1 was 1; Refinement 3 co-tuned to 8 (measured sweet spot)
ROW_BLOCK_TILES = 1  # tile-rows per outer pass; see the assert in the body —
# NOT yet threaded into the compute row-loop (raising it needs a multi-row reduce
# + per-row cb_sumsq expansion). Left at 1 and guarded so it is not a silent
# half-wired knob; a follow-up refinement threads it through.


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>= 1). Keeps W-blocks uniform."""
    k = max(1, min(cap, n))
    while k > 1 and n % k != 0:
        k -= 1
    return k


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _elt(tensor) -> int:
    """Per-element byte size, tolerant of block formats.

    bfloat8_b / bfloat4_b have no well-defined per-element size (16 values share
    one exponent) and `element_size()` raises for them. They are TILE-only, so
    the byte size only ever feeds ROW_MAJOR-stick page math that is never
    allocated for a block-format tensor — return a harmless stand-in of 1.
    """
    try:
        return tensor.element_size()
    except (ValueError, RuntimeError):
        return 1


def _interleaved_assignment(device, total_tile_rows):
    """Row knob-turn for INTERLEAVED: spread the contiguous [0, total_tile_rows)
    tile-row range over the full compute grid via split_work_to_cores (which clips
    to min(cores, work)). Returns (all_cores, [(core, start_tile_row, num_tile_rows)]).
    Each core owns a disjoint contiguous span; the two work groups carry the
    remainder (group 1 gets ceil, group 2 gets floor). No cross-core dependency."""
    grid = device.compute_with_storage_grid_size()
    (_num_cores, all_cores, core_group_1, core_group_2, per_g1, per_g2) = ttnn.split_work_to_cores(
        grid, total_tile_rows, row_wise=True
    )
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, per_g1), (core_group_2, per_g2)):
        if per_core == 0:
            continue
        for c in ttnn.corerange_to_cores(group, None, True):
            assignment.append((c, start, per_core))
            start += per_core
    return all_cores, assignment


def _height_sharded_assignment(input_tensor, total_tile_rows, is_row_major):
    """Row knob-turn for HEIGHT_SHARDED: the SAME contiguous row split, but the
    row->core assignment is pinned by the shard spec instead of split_work_to_cores.

    For TILE the shard's tile-row grid matches the op's work unit exactly:
    `hg = prod(leading)*ceil(H/32)` (eval.sharding) == the op's `total_tile_rows`,
    and each core owns `shard.shape[0]//32` contiguous tile-rows laid out row-major
    (eval.sharding._corerangeset_for_ncores). Iterating the shard grid row-major
    (corerange_to_cores matches that fill order) and striding by the shard height
    hands core i its own shard's tile-rows, so every read/write TensorAccessor
    resolves to that core's local L1 (reduction stays local, no cross-core comms).
    The last shard may be zero-padded, so the last core's `num` is the real
    remainder (< the shard height).

    ROW_MAJOR shards by individual rows (sub-tile granule), so a tile-row can
    straddle a shard boundary — there is no clean tile-row->core match. RM +
    HEIGHT_SHARDED is EXCLUDED op-side, so this path is only reached for TILE; the
    RM branch below is a defensive even split (still correct — TensorAccessor
    routes every page regardless — just not guaranteed core-local)."""
    shard_spec = input_tensor.memory_config().shard_spec
    all_cores = shard_spec.grid
    cores = ttnn.corerange_to_cores(all_cores, None, True)
    if not is_row_major:
        per_core_tiles = int(shard_spec.shape[0]) // TILE_DIM
    else:
        per_core_tiles = math.ceil(total_tile_rows / max(1, len(cores)))
    assignment = []
    start = 0
    for c in cores:
        num = min(per_core_tiles, total_tile_rows - start)
        if num <= 0:
            break
        assignment.append((c, start, num))
        start += per_core_tiles
    return all_cores, assignment


# =====================================================================================
# Refinement 5 — WIDTH_SHARDED + BLOCK_SHARDED cross-core reduction (scheme-change).
# =====================================================================================
# The hidden W is split across a reduction GROUP of cores, so the RMS denominator
# spans core boundaries. Each core reduces its LOCAL W-slice into a partial
# Σx²/W_global (its contribution to the GLOBAL mean(x²)); the reader gathers the
# group's partials onto the group root, the root folds them + rsqrt-finalizes, and
# broadcasts 1/rms back to the group via the mcast_pipe (Mcast2D wire). This is the
# design's dependent-axis scheme-change (references/cross_core_reduction_design.md,
# Pattern A "centralized reduce-root mcast"; transport = mcast_pipe.hpp).
#
# GROUP GEOMETRY (the broadcast-back must reach every member):
#   * BLOCK -> each group is a horizontal core line (grid-row) — always rectangular.
#   * WIDTH -> one group = the whole shard grid (all cores share the tile-rows).
# A RECTANGULAR group (ncores == nx*ny) uses the fast mcast broadcast (Mcast2D wire).
# A RAGGED WIDTH group (ncores != nx*ny — auto_shard_config pads a non-tile-aligned W
# into ceil(W/w_gran) cores, which overflows a full row into a partial one) cannot be
# addressed by a single mcast rectangle. Refinement 5b broadcasts 1/rms to a ragged
# WIDTH group by UNICAST instead (root -> each member individually + a per-member ready
# flag — mirroring the already-unicast gather leg; cross_core_reduction_design.md §8
# option 3). The gather leg and the non-root receiver are topology-agnostic, so only the
# root's broadcast differs (mcast vs unicast). TILE ragged WIDTH still routes to the
# interleaved-collapse fallback (the TILE fallback reads full-W sticks correctly, and
# TILE + w_non is not tile_eligible anyway); the ragged unicast leg is the RM path.


def _sharded_cross_core_plan(input_tensor, total_tile_rows, memory_layout, is_row_major):
    """Derive the per-core cross-core reduction plan from the shard spec.

    Returns (buildable, group_size, assignment, groups, shard_shape, ragged):
      * buildable: a cross-core plan could be built for this geometry (WIDTH always;
        BLOCK when rectangular — BLOCK is always rectangular in practice).
      * assignment: one dict per core — core, start_tile_row, num_tile_rows,
        w_tile_start (TILE), w_col_start (RM gamma col offset), num_w_tiles (local
        W-tiles), my_index (slot in its group), is_root, group_id.
      * groups: one dict per reduction group — rect (CoreRangeSet bbox), root
        (CoreCoord), members (CoreCoord list in slot order — the unicast broadcast
        targets these when ragged).
      * shard_shape: (Hs, Ws) — the uniform per-core shard extent (elements).
      * ragged: the WIDTH group's bbox holds phantom cores (ncores != nx*ny), so the
        broadcast-back must be unicast (R5b), not mcast. Always False for BLOCK.

    R5a: for the ROW_MAJOR leg each core owns a resident [Hs, Ws] shard read
    directly from local L1, so the per-core W-tile count is the ceil-of-shard
    `local_Wt = ceil(Ws/32)` (sub-tile Ws is common — a single zero-padded tile),
    and the per-core tile-row count is `ceil(Hs/32)` (the RM shard flattens the
    leading dims without per-image tile padding). The TILE leg is unchanged
    (whole-tile shards, per-image `total_tile_rows`)."""
    shard = input_tensor.memory_config().shard_spec
    grid = shard.grid
    Hs = int(shard.shape[0])
    Ws = int(shard.shape[1])
    if is_row_major:
        # RM: zero-pad each sub-tile shard slice to whole tiles (single source of truth).
        per_w = math.ceil(Ws / TILE_DIM)  # local W-tiles per core
        rm_tile_rows = math.ceil(Hs / TILE_DIM)  # local tile-rows per core
    else:
        per_h = Hs // TILE_DIM
        per_w = Ws // TILE_DIM
    bbox = grid.bounding_box()
    x0, y0 = int(bbox.start.x), int(bbox.start.y)
    x1, y1 = int(bbox.end.x), int(bbox.end.y)
    nx = x1 - x0 + 1
    ny = y1 - y0 + 1
    ncores = grid.num_cores()
    ragged = ncores != nx * ny
    is_width = memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    # WIDTH: buildable for both rectangular (mcast) and ragged (R5b unicast) grids.
    # BLOCK: only rectangular (grid-row groups are always rectangular; a ragged BLOCK
    # grid is unexpected -> not buildable, caller falls back).
    if per_w < 1 or (ragged and not is_width):
        return False, 0, [], [], (Hs, Ws), ragged

    cores = ttnn.corerange_to_cores(grid, None, True)  # row-major fill order
    assignment = []
    groups = []

    if is_width:
        # One group = every core; H is not split, W split row-major across the grid.
        group_size = ncores
        root = cores[-1]
        rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])
        groups.append({"rect": rect, "root": root, "members": list(cores)})
        num_tile_rows = rm_tile_rows if is_row_major else total_tile_rows
        for i, c in enumerate(cores):
            assignment.append(
                {
                    "core": c,
                    "start_tile_row": 0,
                    "num_tile_rows": num_tile_rows,
                    "w_tile_start": i * per_w,
                    "w_col_start": i * Ws,
                    "num_w_tiles": per_w,
                    "my_index": i,
                    "is_root": (int(c.x) == int(root.x) and int(c.y) == int(root.y)),
                    "group_id": 0,
                }
            )
    else:  # BLOCK_SHARDED — each grid-row is a group (same tile-rows, W split by column).
        group_size = nx
        for gy in range(y0, y1 + 1):
            members = [ttnn.CoreCoord(gx, gy) for gx in range(x0, x1 + 1)]
            root = members[-1]
            rect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, gy), ttnn.CoreCoord(x1, gy))])
            groups.append({"rect": rect, "root": root, "members": members})
            if is_row_major:
                # RM: each core reads its own resident shard from local L1 (start=0),
                # bounded by the uniform shard height Hs.
                st = 0
                num = rm_tile_rows
            else:
                st = (gy - y0) * per_h
                num = max(min(per_h, total_tile_rows - st), 0)
            for idx, c in enumerate(members):
                assignment.append(
                    {
                        "core": c,
                        "start_tile_row": st,
                        "num_tile_rows": num,
                        "w_tile_start": (int(c.x) - x0) * per_w,
                        "w_col_start": (int(c.x) - x0) * Ws,
                        "num_w_tiles": per_w,
                        "my_index": idx,
                        "is_root": (int(c.x) == int(root.x) and int(c.y) == int(root.y)),
                        "group_id": gy - y0,
                    }
                )
    return True, group_size, assignment, groups, (Hs, Ws), ragged


# Cross-core semaphore ids (progress = gather counter; the mcast pipe owns two more).
_SEM_PROGRESS = 0
_SEM_MCAST_READY = 1
_SEM_MCAST_CONSUMED = 2


def _build_cross_core_descriptor(
    input_tensor,
    output_tensor,
    gamma,
    epsilon,
    compute_kernel_config,
    memory_layout,
    origin_H,
    origin_W,
    Wt,
    tiles_per_image,
    total_tile_rows,
    group_size,
    assignment,
    groups,
    is_row_major,
    shard_hw,
    ragged,
):
    """Build the WIDTH/BLOCK cross-core reduce-root program (see the section header).

    R5a: `is_row_major` selects the ROW_MAJOR leg — each core reads/writes its OWN
    resident [Hs, Ws] shard directly from local L1 (the sharded reader/writer local
    legs), compute tilizes/untilizes it (the shared IS_ROW_MAJOR compute branch), and
    the cross-core combine + mcast broadcast are REUSED UNCHANGED. `shard_hw = (Hs, Ws)`
    is the uniform per-core shard extent driving the local-shard stride + padding.

    R5b: `ragged` selects the UNICAST broadcast-back for a non-rectangular WIDTH group.
    The mcast wire (Mcast2D over the bbox) still supplies the semaphore-id / flags CT
    block, but the reader's root leg unicasts 1/rms to each group member (their virtual
    coords ride the root's runtime-arg tail) instead of one mcast to the bbox rectangle
    (which would hit phantom cores). The gather leg and the non-root ReceiverPipe are
    topology-agnostic and unchanged."""
    device = input_tensor.device()
    has_gamma = gamma is not None
    gamma_is_row_major = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT
    gamma_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT
    shard_h, shard_w = int(shard_hw[0]), int(shard_hw[1])

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    gamma_dtype = gamma.dtype if has_gamma else in_dtype
    interm_dtype = ttnn.bfloat16 if in_dtype == ttnn.bfloat8_b else in_dtype
    # R5c: RM cross-core + TILE gamma (fp32/bf16). Each core owns a SUB-TILE global
    # W-column offset (w_col_start = i*Ws), so a TILE-stored gamma can't be read as whole
    # tiles aligned to local col 0. The reader extracts the containing gamma tile(s)' row-0
    # sub-columns into cb_gamma_rm (face-aware L1 byte copy), then the SAME RM-gamma compute
    # tilize leg produces cb_gamma_tiles — so the gamma is "fed via RM" even though it is
    # TILE-stored. bf8b TILE gamma is EXCLUDED op-side (block-float sub-tile extraction needs
    # an in-reader dequant — filed as Refinement 5d).
    gamma_tile_extract = is_row_major and gamma_is_tile and gamma_dtype in (ttnn.float32, ttnn.bfloat16)
    gamma_via_rm = gamma_is_row_major or gamma_tile_extract
    # Partials / gathered stats / broadcast 1/rms are fp32 so the cross-core sum
    # does not truncate (same rationale as the R1/R2a AccumulateViaAdd accumulator).
    stat_dtype = ttnn.float32
    # Gamma-via-RM (RM gamma, or the R5c TILE-extract leg) is tilized by compute, so its
    # tiles carry interm precision; a direct TILE read keeps the on-disk gamma dtype.
    gamma_tiles_dtype = gamma_dtype if (has_gamma and not gamma_via_rm) else interm_dtype

    input_elt = _elt(input_tensor)
    gamma_elt = _elt(gamma) if has_gamma else input_elt

    # LOCAL W-block geometry: every core owns `num_w_tiles` W-tiles (uniform per
    # scheme). Derive the block factor from that local count (single source of truth).
    num_w_tiles = assignment[0]["num_w_tiles"]
    W_BLOCK_TILES = _largest_divisor_leq(num_w_tiles, W_BLOCK_TARGET)
    num_w_blocks = num_w_tiles // W_BLOCK_TILES

    scaler_bits = _f32_bits(1.0 / float(origin_W))  # 1/W_global (the mean divisor)
    eps_bits = _f32_bits(epsilon)

    in_tile = ttnn.tile_size(in_dtype)
    out_tile = ttnn.tile_size(out_dtype)
    interm_tile = ttnn.tile_size(interm_dtype)
    stat_tile = ttnn.tile_size(stat_dtype)
    gamma_tiles_tile = ttnn.tile_size(gamma_tiles_dtype)
    scaler_tile = ttnn.tile_size(ttnn.bfloat16)
    wblock_cols = W_BLOCK_TILES * TILE_DIM
    gamma_rm_page = wblock_cols * gamma_elt

    all_cores = input_tensor.memory_config().shard_spec.grid

    # --- CB indices (semantic; 27/28/29 are the cross-core additions) ---
    CB_INPUT_RM = 0
    CB_INPUT_TILES = 1
    CB_SCALER = 2
    CB_GAMMA_RM = 3
    CB_GAMMA_TILES = 4
    CB_GAMMA_SRC = 5  # R5c: scratch for the containing global gamma tile(s)
    CB_OUTPUT_TILES = 16
    CB_OUTPUT_RM = 17
    CB_XSQ = 24
    CB_SUMSQ = 25
    CB_NORM = 26
    CB_PARTIAL = 27
    CB_GATHER = 28
    CB_RMS_SRC = 29
    CB_COMBINE = 30

    double = 2
    in_rm_page = wblock_cols * input_elt  # RM leg: one W-block-wide local stick slice
    out_rm_page = out_tile  # untilize emits tile-sized pages

    def cb(index, dtype, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        cb(CB_INPUT_TILES, in_dtype, in_tile, double * W_BLOCK_TILES),
        cb(CB_SCALER, ttnn.bfloat16, scaler_tile, 1),
        cb(CB_OUTPUT_TILES, out_dtype, out_tile, double * W_BLOCK_TILES),
        cb(CB_XSQ, interm_dtype, interm_tile, W_BLOCK_TILES),
        # local partial Σx²/W_global (pass-1 accumulator, fp32)
        cb(CB_PARTIAL, stat_dtype, stat_tile, 2),
        # gathered group partials on the root (fp32); one round deep so the write
        # pointer is stable across tile-rows for the peers' unicast targeting
        cb(CB_GATHER, stat_dtype, stat_tile, group_size),
        # broadcast 1/rms landing CB (fp32); depth 1 so the mcast dst offset is
        # uniform across every group member (the mcast_pipe precondition)
        cb(CB_SUMSQ, stat_dtype, stat_tile, 1),
        # root's finalized 1/rms, source of the broadcast (fp32); ONE clean push per
        # tile-row (the reader waits on it, so it must not carry fold intermediates)
        cb(CB_RMS_SRC, stat_dtype, stat_tile, 2),
        # root's combine accumulator (churns per fold block; compute-internal, fp32)
        cb(CB_COMBINE, stat_dtype, stat_tile, 2),
    ]
    if is_row_major:
        # R5a: RM leg — local shard sticks -> compute tilize; compute untilize -> local shard.
        cbs.append(cb(CB_INPUT_RM, in_dtype, in_rm_page, double * TILE_DIM))
        cbs.append(cb(CB_OUTPUT_RM, out_dtype, out_rm_page, double * W_BLOCK_TILES))
    if has_gamma:
        if gamma_via_rm:
            # RM gamma / R5c TILE-extract leg both feed a per-core gamma stick into
            # cb_gamma_rm (compute tilizes it). The real-RM leg reads a column-slice from a
            # 32B-aligned DRAM base, so add DRAM_ALIGN(32B) of page slack for the aligned-read
            # overshoot (shift-copy places the slice at local col 0). The extract leg fills
            # cb_gamma_rm by L1 copy (no DRAM sub-align), so the slack is unused there.
            gamma_rm_alloc = gamma_rm_page + (32 if is_row_major else 0)
            cbs.append(cb(CB_GAMMA_RM, gamma_dtype, gamma_rm_alloc, double))
        if gamma_tile_extract:
            # R5c scratch: hold the up-to-(W_BLOCK_TILES+1) containing global gamma tiles a
            # W-block's sub-tile slice can span (aligned span W_BLOCK_TILES + 1 for a
            # misaligned start). Reader-local (read whole tiles -> extract row 0 -> pop).
            gamma_src_tile = ttnn.tile_size(gamma_dtype)
            cbs.append(cb(CB_GAMMA_SRC, gamma_dtype, gamma_src_tile, W_BLOCK_TILES + 1))
        cbs.append(cb(CB_GAMMA_TILES, gamma_tiles_dtype, gamma_tiles_tile, double * W_BLOCK_TILES))
        cbs.append(cb(CB_NORM, interm_dtype, interm_tile, W_BLOCK_TILES))

    # --- mcast broadcast wire: one Mcast2D per reduction group (root -> group rect) ---
    mcast_cfg = ttnn.McastConfig(sem_ids=[_SEM_MCAST_READY, _SEM_MCAST_CONSUMED])
    helpers = [ttnn.Mcast2D(device, g["rect"], g["root"], mcast_cfg) for g in groups]
    mcast_ct = list(helpers[0].compile_time_args())
    # group_id -> its Mcast2D helper (per-core RT is read off the owning group's helper)
    helper_by_group = {i: helpers[i] for i in range(len(groups))}

    semaphores = [
        ttnn.SemaphoreDescriptor(id=_SEM_PROGRESS, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_MCAST_READY, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=_SEM_MCAST_CONSUMED, core_ranges=all_cores, initial_value=0),
    ]

    input_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    output_addr = output_tensor.buffer_address()

    # ================= Reader (sharded cross-core) =================
    reader_ct_args = [
        1 if has_gamma else 0,
        1 if is_row_major else 0,  # IS_ROW_MAJOR (R5a): local-shard RM read leg
        scaler_bits,
        origin_W,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        gamma_elt,
        1 if gamma_is_row_major else 0,
        group_size,
        _SEM_PROGRESS,
        shard_h,  # RM leg: local shard rows (Hs)
        shard_w,  # RM leg: local shard cols (Ws)
        input_elt,  # RM leg: input element bytes (local shard stride)
        1 if ragged else 0,  # USE_UNICAST_BCAST (R5b): ragged WIDTH group -> unicast 1/rms
        1 if gamma_tile_extract else 0,  # GAMMA_TILE_EXTRACT (R5c): extract TILE gamma row-0 sub-cols
    ]
    reader_ct_args.extend(mcast_ct)  # McastArgs block (CT base = 16)
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # R5b ragged unicast broadcast: the root reaches each group member individually, so it
    # needs their VIRTUAL coords in slot order. Precompute per group (member slot i lines up
    # with my_index=i / w_col_start=i*Ws). Only appended to the ragged root's RT tail; the
    # mcast (rectangular) path never reads them.
    group_member_virt = []
    for g in groups:
        mv = []
        for m in g.get("members", []):
            v = device.worker_core_from_logical_core(m)
            mv.extend([int(v.x), int(v.y)])
        group_member_virt.append(mv)

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    for a in assignment:
        c = a["core"]
        root = groups[a["group_id"]]["root"]
        root_v = device.worker_core_from_logical_core(root)
        mcast_rt = list(helper_by_group[a["group_id"]].runtime_args(c))
        reader_rt = [
            input_addr,
            gamma_addr,
            a["start_tile_row"],
            a["num_tile_rows"],
            a["w_tile_start"],
            a["my_index"],
            1 if a["is_root"] else 0,
            int(root_v.x),
            int(root_v.y),
            a["w_col_start"],  # RM leg: this core's global W-col start (gamma slice)
        ] + mcast_rt
        # Ragged WIDTH root: append the group's member virtual coords (2 words each) for the
        # unicast broadcast. Reader reads them at mc.next_runtime_args_offset() (= RT 14).
        if ragged and a["is_root"]:
            reader_rt = reader_rt + group_member_virt[a["group_id"]]
        reader_rt_args[c.x][c.y] = reader_rt
        writer_rt_args[c.x][c.y] = [output_addr, a["start_tile_row"], a["num_tile_rows"], a["w_tile_start"]]
        compute_rt_args[c.x][c.y] = [a["num_tile_rows"], a["start_tile_row"], eps_bits, 1 if a["is_root"] else 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_sharded_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ================= Writer (shared kernel; TILE global-page or RM local-shard) =================
    writer_ct_args = [
        1 if is_row_major else 0,  # IS_ROW_MAJOR (R5a: RM local-shard writeback leg)
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        _elt(output_tensor),
        1,  # IS_CROSS_CORE (selects the local-shard writeback on the RM leg)
        shard_h,
        shard_w,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ================= Compute (shared kernel; IS_CROSS_CORE branch) =================
    compute_ct_args = [
        1 if is_row_major else 0,  # IS_ROW_MAJOR (R5a: tilize/untilize the local shard)
        1 if has_gamma else 0,
        0,  # HAS_PARTIAL_W (zero-padded W-tail covers a non-aligned W; no partial scaler)
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        scaler_bits,  # 1/W_global (the mean divisor, carried by the pass-1 scaler CB)
        # USE_ACC_VIA_ADD = 0: pass 1 uses ReduceTile (matmul-with-ones), which yields
        # col-0-ONLY partials so the cross-core combine's column-collapse is idempotent.
        # (AccumulateViaAdd leaves the other columns unspecified -> a scale bug in the
        # combine.) The local W-slice is small (<=W_BLOCK_TILES tiles), so ReduceTile's
        # ∝W-slice bias is negligible vs the bf16 tolerance.
        0,
        # GAMMA_IS_ROW_MAJOR (compute): tilize cb_gamma_rm -> cb_gamma_tiles. Set for a real
        # RM gamma AND for the R5c TILE-extract leg (both feed a per-core gamma stick).
        1 if gamma_via_rm else 0,
        1,  # IS_CROSS_CORE
        group_size,
    ]
    compute_config = compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor()
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=cbs,
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.ProgramDescriptor:
    shape = list(input_tensor.shape)
    origin_H = int(shape[-2])
    origin_W = int(shape[-1])
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)

    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    gamma_is_row_major = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    # --- Derived geometry (all from the knobs + shape) ---
    Wt = math.ceil(origin_W / TILE_DIM)
    tiles_per_image = math.ceil(origin_H / TILE_DIM)
    total_tile_rows = leading * tiles_per_image
    # Refinement 3 co-tune: pick the largest divisor of Wt that is <= the target
    # so every W-block is uniformly W_BLOCK_TILES tiles (no partial last W-block).
    # This is the single source of truth for the block factor; every dependent
    # (CB page counts, loop trip counts, num_w_blocks) derives from it below.
    W_BLOCK_TILES = _largest_divisor_leq(Wt, W_BLOCK_TARGET)
    assert Wt % W_BLOCK_TILES == 0, "W_BLOCK_TILES must divide Wt (holds by construction)"
    num_w_blocks = Wt // W_BLOCK_TILES
    # ROW_BLOCK_TILES>1 would need a multi-row reduce + per-row cb_sumsq expansion
    # in the compute row-loop, which is not implemented; guard so it is explicit.
    assert ROW_BLOCK_TILES == 1, "ROW_BLOCK_TILES>1 not yet threaded into the compute row-loop"

    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0
    scaler_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    input_elt = _elt(input_tensor)
    output_elt = _elt(output_tensor)
    gamma_elt = _elt(gamma) if has_gamma else input_elt

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    gamma_dtype = gamma.dtype if has_gamma else in_dtype

    # Intermediate (accumulator/scratch) CB format. fp32 input keeps fp32
    # intermediates (fp32 dest-acc path); bf16 keeps bf16; bf8b uses bf16 — a
    # block-float accumulator/scratch would be far too lossy for Sum(x^2) and the
    # x*(1/rms) scratch. This is identity for fp32/bf16 (byte-identical to prior
    # passing cells) and only lifts bf8b's intermediates off the block format.
    interm_dtype = ttnn.bfloat16 if in_dtype == ttnn.bfloat8_b else in_dtype

    # REDUCE DATAPATH selector (Refinement 1 for fp32; Refinement 2a extends it to
    # bf16). The tile-aligned float Σx² reduce runs on ReduceAlgorithm::AccumulateViaAdd:
    # the accumulator holds the RAW element-wise Σx² tile (folded per W-block with
    # add_tiles) and is reduced ONCE on the last block — removing the per-block
    # reduced-partial reload of ReduceTile whose truncation undercounts mean(x²) ∝ W.
    #   * fp32 (R1): the accumulator was already fp32 (interm_dtype), fixed the
    #     ∝W scale bias.
    #   * bf16 (R2a): the accumulator was bf16 and hit a catastrophic cliff at very
    #     wide W (W=32768: rel-RMS 0.40). Extending the AccumulateViaAdd datapath
    #     alone is NOT enough — the RAW running sum must not truncate, so the
    #     accumulator CB is forced to fp32 here (the reduce helper natively folds a
    #     bf16 input CB into an fp32 accumulator: reconfig_data_format_srcb/srca
    #     around the acc-add, per reduce_helpers_compute.inl).
    # bf8b stays on ReduceTile (already passes there, R2) and the non-tile-aligned
    # partial path stays on ReduceTile (AccumulateViaAdd cross-call cannot express
    # the masked partial tile). This is the R2 null-result's real fix: R2 measured
    # that merely forcing cb_sumsq fp32 on the *ReduceTile* path was a net regression
    # (removed the cliff but exposed the smooth ∝W bias); the fix is the fp32
    # accumulator ON the AccumulateViaAdd datapath, which has no ∝W bias.
    use_acc_via_add = in_dtype in (ttnn.float32, ttnn.bfloat16) and not has_partial_w
    # Accumulator CB format: fp32 whenever AccumulateViaAdd is used (raw Σx² must
    # not truncate); otherwise interm_dtype (unchanged ReduceTile path). cb_xsq
    # stays interm_dtype — it holds individual x² values (small; bf16 is fine) and
    # the helper handles the mixed bf16-input / fp32-accumulator fold.
    sumsq_dtype = ttnn.float32 if use_acc_via_add else interm_dtype

    # Gamma tiles: TILE gamma is a raw tile copy (reader writes the on-disk bytes,
    # so the CB MUST carry gamma_dtype); RM gamma is tilized by compute (which
    # converts format), so pack it at the intermediate precision (== in_dtype for
    # fp32/bf16 — unchanged; bf16 for bf8b input).
    gamma_tiles_dtype = gamma_dtype if (has_gamma and not gamma_is_row_major) else interm_dtype

    in_tile = ttnn.tile_size(in_dtype)
    out_tile = ttnn.tile_size(out_dtype)
    interm_tile = ttnn.tile_size(interm_dtype)
    sumsq_tile = ttnn.tile_size(sumsq_dtype)
    gamma_tiles_tile = ttnn.tile_size(gamma_tiles_dtype)
    scaler_tile = ttnn.tile_size(ttnn.bfloat16)

    wblock_cols = W_BLOCK_TILES * TILE_DIM
    in_rm_page = wblock_cols * input_elt  # one W-block-wide stick slice
    out_rm_page = out_tile  # untilize emits tile-sized pages
    gamma_rm_page = wblock_cols * gamma_elt

    # --- Grid + per-core work assignment (Refinement 4: Row-axis knob-turn) ---
    # The Row axis is INDEPENDENT (each tile-row's RMS is computed in isolation, no
    # cross-row dependency), so distributing tile-rows across the grid needs NO
    # cross-core communication — the kernels already key off per-core
    # (start_tile_row, num_tile_rows) RT args and iterate `start_tile_row + t`, so
    # this is a pure runtime-arg change (no loop-nest / kernel change).
    #
    #   * INTERLEAVED  — `split_work_to_cores(grid, total_tile_rows)` spreads the
    #                    contiguous tile-row range over the full compute grid.
    #   * HEIGHT_SHARDED — the SAME row split, but the row->core assignment is pinned
    #                    by the shard spec (each core owns shard.shape[0]//32 tile-rows)
    #                    instead of split_work_to_cores. The reduction stays LOCAL per
    #                    core; TensorAccessor (built from the sharded tensor's
    #                    memory-config) routes each global page id to the owning core's
    #                    L1, so the resident shard is streamed through the SAME bounded
    #                    scratch CBs (a local L1->L1 read). We deliberately do NOT point
    #                    the input/output CBs at the whole resident shard via
    #                    cb_descriptor_from_sharded_tensor: that would force a
    #                    random-access rewrite of the two-pass streaming compute (a
    #                    consumed CB cannot be rewound for pass 2) and a Wt-sized CB,
    #                    breaking the design's bounded-streaming invariant — a
    #                    scheme-change, not the Row knob-turn this refinement is.
    device = input_tensor.device()
    memory_layout = input_tensor.memory_config().memory_layout
    is_height_sharded = memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    is_width_or_block = memory_layout in (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )

    # R5 cross-core scheme-change (WIDTH/BLOCK): the W is split across cores, so the
    # reduction spans core boundaries. Route to the reduce-root mcast program when the
    # group geometry is mcastable (TILE input, tile-aligned W, rectangular shard grid);
    # otherwise fall through to the interleaved streaming path below (a correct,
    # verified fallback: one core streams the whole W of each tile-row from the
    # resident shards via TensorAccessor — no cross-core sync, no hang).
    # R5a extends the cross-core route to ROW_MAJOR input. RM width-sharding splits each
    # logical row's W across cores at sub-tile (stick) granularity, so a row is not
    # contiguous in any one core's L1; each core instead reads/writes its OWN resident
    # [Hs, Ws] shard from local L1, zero-pads the sub-tile W (and H) tail to whole tiles,
    # and the SAME reduce-root gather + mcast broadcast combine runs unchanged. All RM
    # cross-core groups are rectangular (BLOCK grids always are; {RM, WIDTH, w_non} — the
    # only ragged RM geometry — is op-side EXCLUDED), so the mcast transport is reused.
    if is_width_or_block:
        buildable, group_size, cc_assignment, cc_groups, cc_shard_hw, ragged = _sharded_cross_core_plan(
            input_tensor, total_tile_rows, memory_layout, is_row_major
        )
        # TILE cross-core (R5): tile-aligned W + rectangular; else -> interleaved fallback
        # (a ragged/non-aligned TILE grid streams full-W sticks correctly there).
        # RM cross-core (R5a rectangular = mcast; R5b ragged WIDTH = unicast broadcast).
        # No partial-W gate for RM — the zero-padded W-tail handles a non-aligned W with
        # the same 1/W_global scaler (no partial scaler).
        tile_eligible = (not is_row_major) and (not has_partial_w)
        # Route to the cross-core reduce-root program when buildable and the group is
        # addressable: TILE needs a rectangular group (mcast only); RM accepts a ragged
        # WIDTH group too (R5b unicast broadcast). Ragged TILE falls back below.
        addressable = is_row_major or not ragged
        if (is_row_major or tile_eligible) and buildable and group_size > 1 and addressable:
            return _build_cross_core_descriptor(
                input_tensor,
                output_tensor,
                gamma,
                epsilon,
                compute_kernel_config,
                memory_layout,
                origin_H,
                origin_W,
                Wt,
                tiles_per_image,
                total_tile_rows,
                group_size,
                cc_assignment,
                cc_groups,
                is_row_major,
                cc_shard_hw,
                ragged,
            )
        if is_row_major:
            # Every RM WIDTH/BLOCK geometry is now buildable (rectangular -> mcast, ragged
            # WIDTH -> unicast). This guard is defensive — reached only if buildable is False
            # (per_w < 1, structurally impossible for RM), never for a supported cell.
            raise NotImplementedError(
                "rms_norm: ROW_MAJOR WIDTH/BLOCK sharding could not build a reduction group "
                f"(got group_size={group_size}, buildable={buildable}, ragged={ragged})"
            )

    if is_height_sharded:
        all_cores, assignment = _height_sharded_assignment(input_tensor, total_tile_rows, is_row_major)
    else:
        # INTERLEAVED, or the WIDTH/BLOCK fallback (ragged/RM/non-aligned): stream the
        # full W of each assigned tile-row via TensorAccessor over the shard grid.
        all_cores, assignment = _interleaved_assignment(device, total_tile_rows)

    # --- CB indices (semantic) ---
    CB_INPUT_RM = 0
    CB_INPUT_TILES = 1
    CB_SCALER = 2
    CB_GAMMA_RM = 3
    CB_GAMMA_TILES = 4
    CB_OUTPUT_TILES = 16
    CB_OUTPUT_RM = 17
    CB_XSQ = 24
    CB_SUMSQ = 25
    CB_NORM = 26

    # --- Buffer-depth knobs (data-movement<->compute overlap, not reuse) ---
    STICK_BLOCK = TILE_DIM  # one tile-row height of sticks
    double = 2

    def cb(index, dtype, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        # streamed input tiles (reader in TILE regime / tilize in RM regime)
        cb(CB_INPUT_TILES, in_dtype, in_tile, double * W_BLOCK_TILES),
        # reduce scaler (bf16): full [+ partial]
        cb(CB_SCALER, ttnn.bfloat16, scaler_tile, 2 if has_partial_w else 1),
        # normalized output tiles (mul -> writer in TILE / -> untilize in RM)
        cb(CB_OUTPUT_TILES, out_dtype, out_tile, double * W_BLOCK_TILES),
        # x^2 scratch (pass 1) — compute->compute, single-depth full block
        cb(CB_XSQ, interm_dtype, interm_tile, W_BLOCK_TILES),
        # Sum(x^2)/W accumulator -> 1/rms (held across pass 2). fp32 on the
        # AccumulateViaAdd path (raw Σx² must not truncate); interm_dtype otherwise.
        cb(CB_SUMSQ, sumsq_dtype, sumsq_tile, max(double * ROW_BLOCK_TILES, 2)),
    ]
    if is_row_major:
        # RM input sticks -> tilize; RM output sticks <- untilize
        cbs.append(cb(CB_INPUT_RM, in_dtype, in_rm_page, double * STICK_BLOCK))
        cbs.append(cb(CB_OUTPUT_RM, out_dtype, out_rm_page, double * W_BLOCK_TILES))
    if has_gamma:
        # cb_gamma_rm only exists on the RM-gamma leg (reader sticks -> compute
        # tilize). TILE gamma skips it: the reader writes tiles straight into
        # cb_gamma_tiles (single producer per build — the two legs are separate
        # compiled programs, so the one-producer rule holds per build).
        if gamma_is_row_major:
            cbs.append(cb(CB_GAMMA_RM, gamma_dtype, gamma_rm_page, double))
        cbs.append(cb(CB_GAMMA_TILES, gamma_tiles_dtype, gamma_tiles_tile, double * W_BLOCK_TILES))
        # normalize scratch (x*(1/rms)) before gamma mul — compute->compute
        cbs.append(cb(CB_NORM, interm_dtype, interm_tile, W_BLOCK_TILES))

    # ================= Reader =================
    reader_ct_args = [
        1 if is_row_major else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        scaler_bits,
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        input_elt,
        gamma_elt,
        # RM gamma -> stick-read + compute tilize; TILE gamma -> read tiles direct.
        1 if gamma_is_row_major else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    input_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    output_addr = output_tensor.buffer_address()

    reader_rt_args = ttnn.RuntimeArgs()
    for c, c_start, c_num in assignment:
        reader_rt_args[c.x][c.y] = [input_addr, gamma_addr, c_start, c_num]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ================= Writer =================
    writer_ct_args = [
        1 if is_row_major else 0,
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        output_elt,
        0,  # IS_CROSS_CORE (interleaved/HEIGHT use TensorAccessor global-stick addressing)
        0,  # shard_h (unused off the cross-core RM leg)
        0,  # shard_w (unused off the cross-core RM leg)
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    for c, c_start, c_num in assignment:
        # 4th arg = w_tile_start; 0 here (this path owns the full W of each tile-row).
        writer_rt_args[c.x][c.y] = [output_addr, c_start, c_num, 0]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ================= Compute =================
    compute_ct_args = [
        1 if is_row_major else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        # 1/W as float bits: the mean scaler. For the tile-aligned AccumulateViaAdd
        # reduce path (SUM has no scaler tile) this is applied as the last-chunk
        # post_reduce_op; the ReduceTile path applies it via the bf16 scaler CB instead.
        scaler_bits,
        # USE_ACC_VIA_ADD selects the accurate AccumulateViaAdd reduce datapath
        # (R1 for fp32, R2a extends it to bf16 with an fp32 accumulator CB). Already
        # folds in !has_partial_w. bf8b and the non-tile-aligned partial path keep
        # the unchanged ReduceTile path.
        1 if use_acc_via_add else 0,
        # GAMMA_IS_ROW_MAJOR: RM gamma is tilized by compute; TILE gamma arrives
        # already tiled from the reader (skip the gamma-tilize step).
        1 if gamma_is_row_major else 0,
        # IS_CROSS_CORE, GROUP_SIZE — 0 here (this path finalizes 1/rms locally; the
        # WIDTH/BLOCK reduce-root program sets these in _build_cross_core_descriptor).
        0,
        0,
    ]
    compute_rt_args = ttnn.RuntimeArgs()
    for c, c_start, c_num in assignment:
        # 4th arg = is_root; unused off the cross-core path.
        compute_rt_args[c.x][c.y] = [c_num, c_start, eps_bits, 0]
    # compute_kernel_config is threaded through as-is (math_fidelity /
    # fp32_dest_acc_en / math_approx_mode / dst_full_sync_en all honored by the
    # kernel's helpers). No CB is tagged UnpackToDestFp32: every fp32 intermediate
    # here (cb_xsq, cb_sumsq) feeds an FPU op — the reduce, or the AccumulateViaAdd
    # add_tiles fold — and UnpackToDestFp32 is exclusive with any FPU consumer
    # (numeric-formats-metal §1.5). So no CB qualifies; tagging would break math.
    compute_config = compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor()
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
