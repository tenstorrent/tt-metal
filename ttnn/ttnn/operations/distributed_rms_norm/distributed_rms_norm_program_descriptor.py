# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — plan (groups, links, split, records) and the MeshProgramDescriptor.

Regime R1 (op_design.md): interleaved x resident in L1, rows x W-chunks over the worker rectangle, flat fp32
record all-gather over Fabric through port cores in logical row 0.
Regimes R9 / R10 (physical shards): the worker set is the shard grid and each worker's block is its shard,
consumed in place (cb_x_resident is backed on the input shard, cb_output_tiles on the output shard when the output
shard spec matches). HEIGHT pins S_w = 1; WIDTH pins S_r = 1 and S_w = shard cores, and adds the R5 device-local
pre-combine (each W chunk publishes into a combiner's inbox; combiners publish one record per tile row), because a
pinned S_w of up to ~86 makes flat records quadratic. Ports sit on grid cores the shard grid leaves free.

Per device:
  ports   : interleaved: logical (2*l + d, 0); sharded: free cores. l < L_eff, d = 0 (fwd, toward p+1) /
            1 (bwd, toward p-1). BRISC, NoC0 (PORT_NOC).
  workers : interleaved: logical rows 1.., row-major index k = r_g * S_w + w; sharded: the shard cores.
            reader NCRISC/NoC0, writer BRISC/NoC1, compute.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- circular buffers (worker cores) ---------------------------------------------------------------------------
CB_X_RESIDENT = 0
CB_GAMMA_TILES = 1
CB_SCALER = 2
CB_SQ_ACC = 3
CB_PARTIAL = 4
CB_XG = 5
CB_STATS_COMBINE = 6
CB_RSTD = 7
CB_OUTPUT_TILES = 8
CB_LC = 9  # R5 pre-combine: S_w W-chunk records per tile row (combiners only)
CB_LC_OUT = 10  # R5 pre-combine: one combined record per tile row (combiners only)

# ---- program-local semaphores (declared over every core, before any fabric connection setup) ------------------
ROUND_LOOKAHEAD = 1  # R2: rounds whose load/sumsq/publish run ahead of the oldest round's gather/normalize/store
# on ports: workers of the link increment once per round their records have landed, round b on counter
# SEM_LOCAL_ARRIVALS + b % LOCAL_ARRIVAL_SEMS. A worker publishes round b + 1 + ROUND_LOOKAHEAD only after its round b
# gather, i.e. after every worker of the link published round b; so with 1 + ROUND_LOOKAHEAD rotating counters an
# increment can never stand in for an earlier round's (one shared counter would let a fast worker's round b+1
# increment complete a slow worker's round b).
SEM_LOCAL_ARRIVALS = 0
LOCAL_ARRIVAL_SEMS = 1 + ROUND_LOOKAHEAD
SEM_LOCAL_COMBINE = SEM_LOCAL_ARRIVALS + LOCAL_ARRIVAL_SEMS  # on combiners: W chunks of the row group, once each
# on workers: one counter per port (l, d) at SEM_STATS_READY_BASE + 2*l + d; port (l, d) multicasts +1 per round once
# its round's slots are complete. Per-port (not one shared) counters: with several rounds in flight a fast port's
# round b+1 increment must never stand in for a slow port's round b one.
SEM_STATS_READY_BASE = SEM_LOCAL_COMBINE + 1
NUM_SEMAPHORES = 16  # per core (tt_metal semaphore.hpp)

# ---- blocking knobs (single source of truth) -------------------------------------------------------------------
MIN_CORE_W_TILES = (
    8  # floor of the W chunk per worker; record pulls grow as S_w^2 (perf lamp "grid synchronization": 8 measured best)
)
X_BLOCKS_IN_FLIGHT = 1  # cb_x_resident depth in blocks per round in flight (residency, not pipelining)
EXCHANGE_ROUNDS = 2  # R2: target exchange rounds per call (interleaved); 1 = the single-round R1 schedule
# R5: interleaved gamma read issued with round 0's first x tile row (its single-bank read hides under the x reads);
# False: after the lookahead x blocks (the R3 placement). Measured: True, min device 18.2 -> 17.4 us at 640x1792.
GAMMA_EARLY = True
GAMMA_PRESCALE = True  # x*gamma in the exchange stall (cb_xg) when it fits; False: fused gamma in normalize
OUT_DEPTH = 2  # cb_output_tiles depth in out batches (double buffer)
# R5: port kernels on NoC0. On NoC1 their stats-ready multicasts queue behind every worker's DRAM writes (measured
# 1-3 us from issue to landing in round 1); NoC0 carries only the (finished) x reads and the record pulls then.
PORT_NOC = ttnn.NOC.RISCV_0_default
# R5: interleaved row groups assigned to links in contiguous runs, each port multicasting stats-ready only to its own
# link's workers (False: round-robin links, every port multicasts the whole worker rectangle).
PER_LINK_MCAST = True
HEADER_RING = 8  # fabric packet headers in flight per port (pool holds 12 per RISC)
# R4 knob: tag the Float32 stats CBs read only by copy_tile (the Accurate SFPU reduces' inputs) UnpackToDestFp32, so
# the fp32 partial sums are not truncated to tf32 on the way in (needs a 32-bit DEST). Correct, but PARKED (False):
# measured, it makes the end-to-end error WORSE (fp32 mean ULP 15k -> 19k, bf16 0.12 -> 0.18, median got/true
# 0.9988 -> 0.9984) — the stats' downward truncation was offsetting the downward tf32 truncation of x / gamma / rstd
# on the FPU multiplies. Turn it on together with an fp32-exact multiply path (unpack-to-DEST + SFPU mul).
UNPACK_STATS_TO_DEST = False
STATS_UNPACK_TO_DEST_CBS = (CB_SQ_ACC, CB_STATS_COMBINE, CB_LC)  # never an FPU operand (cb_rstd / cb_xg are)
NUM_CIRCULAR_BUFFERS = 64
BFP8_EXP_BYTES = 64  # Bfp8_b tile: 64 shared exponents (one per face row) ahead of the 1024 mantissa bytes
RECORD_BYTES = 128  # one fp32 record: 32 per-row partial sums of one tile row
WORKER_L1_BUDGET = 1 << 20  # CB bytes per worker (P_resident)
SHARDED_LAYOUTS = (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED)
SCRATCH_BUDGET = 1 << 20  # exchange scratch bytes per port core (P_scratch)
TILE = 32


def block_rows_knob(core_rows_max, rounds):
    """Rows per block: the per-core assignment split into `rounds` blocks (rounds == 1: the whole assignment)."""
    return math.ceil(core_rows_max / rounds)


def blocks_in_flight(num_blocks):
    """Blocks resident at once in the per-block worker CBs (cb_x_resident, cb_xg): the oldest round plus the
    rounds running ahead of it."""
    return min(num_blocks, 1 + ROUND_LOOKAHEAD)


def x_blocks(num_blocks, gamma_prescale):
    """cb_x_resident capacity in blocks. Prescale pops x every round, so x is a plain FIFO of the blocks in flight.
    Otherwise round b's x stays at the CB front until its normalize while round b+1's sumsq reads past it at a tile
    offset; a tile offset is linear from the read pointer and not wrap-safe, so the CB holds the whole assignment
    (it never wraps within a call; the same bytes as the one-round R1 block)."""
    return blocks_in_flight(num_blocks) if gamma_prescale else num_blocks


def out_batch_tiles_knob(core_w_tiles):
    """Tiles per writer barrier (Phase 0: one tile row of the core's W chunk). Must divide core_w_tiles: the writer
    walks each tile row in batches, and cb_output_tiles (OUT_DEPTH batches) must wrap on a batch boundary."""
    return core_w_tiles


# ----------------------------------------------------------------------------------------------------------------
# Groups, links, schedules (shape independent, cached)
# ----------------------------------------------------------------------------------------------------------------


def _groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _wraps_axis(cluster_axis):
    cfg = ttnn.get_fabric_config()
    fc = ttnn.FabricConfig
    wrap0 = {getattr(fc, n, None) for n in ("FABRIC_2D_TORUS_Y", "FABRIC_2D_TORUS_XY")}
    wrap1 = {getattr(fc, n, None) for n in ("FABRIC_2D_TORUS_X", "FABRIC_2D_TORUS_XY")}
    return cfg in (wrap0 if cluster_axis == 0 else wrap1)


def _schedule(p, G, ring):
    """(slots sent toward p+1, slots sent toward p-1), own slot first, in arrival order downstream."""
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [(p - i) % G for i in range(kf)], [(p + i) % G for i in range(kb)] if kb > 0 else []
    fwd = [p - i for i in range(p + 1)] if p < G - 1 else []
    bwd = [p + i for i in range(G - p)] if p > 0 else []
    return fwd, bwd


_LINK_CACHE = {}


def _hop_links(mesh_device, cluster_axis, ring):
    """{(src_coord, dst_coord): [link indices]} for every directed hop of every group (cached)."""
    key = (id(mesh_device), tuple(mesh_device.shape), cluster_axis, ring, str(ttnn.get_fabric_config()))
    if key in _LINK_CACHE:
        return _LINK_CACHE[key][1]
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    hops = {}
    for grp in _groups(tuple(mesh_device.shape), cluster_axis):
        G = len(grp)
        pairs = [(grp[p], grp[p + 1]) for p in range(G - 1)]
        if ring and G > 2:
            pairs.append((grp[G - 1], grp[0]))
        for a, b in pairs:
            hops[(a, b)] = list(ttnn.get_forwarding_link_indices(node(a), node(b)))
            hops[(b, a)] = list(ttnn.get_forwarding_link_indices(node(b), node(a)))
    _LINK_CACHE[key] = (mesh_device, hops)
    return hops


def caller_link_checks(mesh_device, cluster_axis, topology, num_links):
    """ValueError for link / topology requests the live cluster cannot satisfy."""
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    ring = topology == Topology.Ring
    if ring:
        if not _wraps_axis(cluster_axis):
            raise ValueError(
                f"distributed_rms_norm: Ring along cluster_axis {cluster_axis} needs a fabric config that wraps it "
                f"(active: {ttnn.get_fabric_config()})"
            )
        node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
        for grp in _groups(mesh_shape, cluster_axis):
            if not list(ttnn.get_forwarding_link_indices(node(grp[-1]), node(grp[0]))):
                raise ValueError(f"distributed_rms_norm: no wrap link {grp[-1]} -> {grp[0]} for Ring")
    hops = _hop_links(mesh_device, cluster_axis, ring and G > 2)
    min_links = min(len(v) for v in hops.values())
    if min_links == 0:
        raise ValueError("distributed_rms_norm: a hop of the group has no usable fabric link")
    gx = mesh_device.compute_with_storage_grid_size().x
    if num_links is not None and 2 * num_links > gx:
        raise ValueError(
            f"distributed_rms_norm: num_links={num_links} needs {2 * num_links} port columns, grid has {gx}"
        )
    if num_links is not None and num_links > min_links:
        raise ValueError(
            f"distributed_rms_norm: num_links={num_links} exceeds the {min_links} usable link(s) per hop "
            f"(fabric {ttnn.get_fabric_config()})"
        )


# ----------------------------------------------------------------------------------------------------------------
# Split selection and plan
# ----------------------------------------------------------------------------------------------------------------


def _choose_split(tensor_row_tiles, tensor_w_tiles, num_workers, min_core_w_tiles=MIN_CORE_W_TILES, max_s_r=None):
    """(S_r, S_w, core_w_tiles): lexicographically smallest (max tiles per worker, S_w). `max_s_r` caps the row
    groups (R2: every group keeps >= rounds tile rows, so the freed workers go to W chunks)."""
    best = None
    s_r_cap = tensor_row_tiles if max_s_r is None else min(tensor_row_tiles, max_s_r)
    for s_w in range(1, max(1, min(num_workers, math.ceil(tensor_w_tiles / min_core_w_tiles))) + 1):
        s_r = min(s_r_cap, num_workers // s_w)
        if s_r == 0:
            break
        cost = math.ceil(tensor_row_tiles / s_r) * math.ceil(tensor_w_tiles / s_w)
        if best is None or (cost, s_w) < best:
            best = (cost, s_w)
    s_w = best[1]
    core_w_tiles = math.ceil(tensor_w_tiles / s_w)
    s_w = math.ceil(tensor_w_tiles / core_w_tiles)  # no empty trailing chunk
    s_r = min(s_r_cap, num_workers // s_w)
    return s_r, s_w, core_w_tiles


def _tile_bytes(dtype):
    return ttnn.tile_size(dtype)


def _split_cost(tensor_row_tiles, Wt, split):
    S_r, S_w, cw = split
    return math.ceil(tensor_row_tiles / S_r) * cw


def _rounds_split(tensor_row_tiles, Wt, num_workers):
    """(rounds, (S_r, S_w, core_w_tiles)). R2 (op_design.md overlap lamp): EXCHANGE_ROUNDS rounds with S_r capped so
    every row group holds >= rounds tile rows, and the W-chunk floor relaxed so S_w may grow by the same factor as
    S_r shrinks ("S_r = Ht/2, S_w doubled"). Taken only when it does not raise the tiles per worker over the
    one-round split (the overlap hides the exchange; it cannot pay for extra DRAM/compute per worker)."""
    r1 = _choose_split(tensor_row_tiles, Wt, num_workers)
    rounds = max(1, min(EXCHANGE_ROUNDS, tensor_row_tiles))
    if rounds == 1:
        return 1, r1
    floor = min(MIN_CORE_W_TILES, math.ceil(Wt / (rounds * r1[1])))
    r2 = _choose_split(
        tensor_row_tiles, Wt, num_workers, min_core_w_tiles=floor, max_s_r=math.ceil(tensor_row_tiles / rounds)
    )
    if _split_cost(tensor_row_tiles, Wt, r2) > _split_cost(tensor_row_tiles, Wt, r1):
        return 1, r1
    return rounds, r2


def _interleaved_layout(tensor_row_tiles, Wt, gx, gy):
    """R1 / R2: rows x W-chunks over the worker rectangle (logical rows 1..), ports in logical row 0."""
    rounds, (S_r, S_w, cw) = _rounds_split(tensor_row_tiles, Wt, gx * (gy - 1))
    base, extra = divmod(tensor_row_tiles, S_r)
    core_rows = [base + (1 if g < extra else 0) for g in range(S_r)]
    row_start = [sum(core_rows[:g]) for g in range(S_r)]
    n_used = S_r * S_w
    y_last = 1 + (n_used - 1) // gx  # worker rectangle: logical rows 1..y_last, full width
    rect = (ttnn.CoreCoord(0, 1), ttnn.CoreCoord(gx - 1, y_last))
    workers = []
    for k in range(n_used):
        g, w = k // S_w, k % S_w
        cw_here = min(cw, Wt - w * cw)
        workers.append(
            dict(
                core=ttnn.CoreCoord(k % gx, 1 + k // gx),
                g=g,
                w=w,
                row_start=row_start[g],
                rows=core_rows[g],
                w_start=w * cw,
                cw=cw_here,
                cw_phys=cw_here,
            )
        )
    worker_cores = [ttnn.CoreCoord(x, y) for y in range(1, y_last + 1) for x in range(gx)]
    return dict(
        rounds=rounds,
        S_r=S_r,
        S_w=S_w,
        core_w_tiles=cw,
        core_rows=core_rows,
        row_start=row_start,
        workers=workers,
        worker_cores=worker_cores,
        worker_set=ttnn.CoreRangeSet([ttnn.CoreRange(*rect)]),
        mcast_rects=[rect],
        free_cores=None,  # ports: logical row 0
    )


def _linear_rects(cores, gx):
    """Rectangles (start, end) covering exactly `cores`, a row-major run of the worker grid (full width gx): a
    partial first row, the full rows, a partial last row."""
    if not cores:
        return []
    idx = sorted(c.y * gx + c.x for c in cores)
    assert idx == list(range(idx[0], idx[-1] + 1)), "a link's workers are one row-major run"
    a, b = idx[0], idx[-1] + 1
    rects = []
    while a < b:
        y, x0 = divmod(a, gx)
        if x0 == 0 and b - a >= gx:
            rows = (b - a) // gx
            rects.append((ttnn.CoreCoord(0, y), ttnn.CoreCoord(gx - 1, y + rows - 1)))
            a += rows * gx
        else:
            x1 = min(gx, x0 + (b - a))
            rects.append((ttnn.CoreCoord(x0, y), ttnn.CoreCoord(x1 - 1, y)))
            a += x1 - x0
    return rects


def _sharded_layout(mem_config, tensor_row_tiles, Wt, gx, gy):
    """R9 / R10: the shard grid is the worker set and the shard is the block (op_design.md)."""
    spec = mem_config.shard_spec
    sh, sw = int(spec.shape[0]), int(spec.shape[1])
    assert sh % TILE == 0 and sw % TILE == 0, "TILE shards are whole tiles"
    rows_per, w_per = sh // TILE, sw // TILE
    cores = list(ttnn.corerange_to_cores(spec.grid, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))
    workers = []
    if mem_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        assert w_per == Wt, "HEIGHT shard holds the full W_local"
        for i, core in enumerate(cores):
            rows = max(0, min(rows_per, tensor_row_tiles - i * rows_per))
            if rows:
                workers.append(
                    dict(
                        core=core,
                        g=len(workers),
                        w=0,
                        row_start=i * rows_per,
                        rows=rows,
                        w_start=0,
                        cw=Wt,
                        cw_phys=w_per,
                    )
                )
        core_rows = [wk["rows"] for wk in workers]
        row_start = [wk["row_start"] for wk in workers]
        S_r, S_w = len(workers), 1
    else:
        assert rows_per >= tensor_row_tiles, "WIDTH shard holds every tile row"
        for j, core in enumerate(cores):
            cw = max(0, min(w_per, Wt - j * w_per))
            if cw:
                workers.append(
                    dict(
                        core=core,
                        g=0,
                        w=len(workers),
                        row_start=0,
                        rows=tensor_row_tiles,
                        w_start=j * w_per,
                        cw=cw,
                        cw_phys=w_per,
                    )
                )
        core_rows, row_start = [tensor_row_tiles], [0]
        S_r, S_w = 1, len(workers)
    taken = {(c.x, c.y) for c in cores}
    free = [ttnn.CoreCoord(x, y) for y in range(gy) for x in range(gx) if (x, y) not in taken]
    rects = [(r.start, r.end) for r in spec.grid.ranges()]
    return dict(
        rounds=1,  # the shard is the block; R5 pre-combine runs once
        S_r=S_r,
        S_w=S_w,
        core_w_tiles=w_per if S_w > 1 else Wt,
        core_rows=core_rows,
        row_start=row_start,
        workers=workers,
        worker_cores=cores,
        worker_set=spec.grid,
        mcast_rects=rects,
        free_cores=free,
    )


def xg_dtype_of(fp32_dest_acc_en):
    """cb_xg (x*gamma, a DEST-produced intermediate) follows the DEST width: a Float32 page stores no extra bits
    when DEST is 16-bit. The stats CBs and wire records stay Float32 regardless (the spec's fp32 exchange)."""
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def _fit_rounds(lay, x_dtype, gamma_dtype, n_src_tiles, budget=WORKER_L1_BUDGET, xg_dtype=ttnn.float32):
    """(rounds, prescale_fits) for the interleaved path: R2 also lifts the P_resident ceiling. With 1 + ROUND_LOOKAHEAD
    blocks in flight, more rounds shrink the resident set; take the fewest rounds (>= the split's) whose footprint
    fits, preferring the gamma-prescale path (cb_xg) over the fused-gamma fallback (measured ~2x slower)."""
    core_rows_max = max(lay["core_rows"])

    def plan_at(rounds):
        br = block_rows_knob(core_rows_max, rounds)
        return dict(
            block_rows=br,
            num_blocks=max(math.ceil(r / br) for r in lay["core_rows"]),
            core_w_tiles=lay["core_w_tiles"],
            sharded=False,
            n_src_tiles=n_src_tiles,
            local_combine=False,
            lc_per=[0],
            xg_dtype=xg_dtype,
        )

    prefs = [True, False] if gamma_dtype is not None and GAMMA_PRESCALE else [False]
    for prescale in prefs:
        for rounds in range(lay["rounds"], core_rows_max + 1):
            if _worker_footprint(plan_at(rounds), x_dtype, gamma_dtype, prescale) <= budget:
                return rounds
    return lay["rounds"]  # nothing fits: run_distributed_rms_norm's footprint check reports it


def build_plan(input_tensor, *, cluster_axis, topology, num_links, gamma_dtype=None, fp32_dest_acc_en=True):
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    ring = topology == Topology.Ring and mesh_shape[cluster_axis] > 2
    G = mesh_shape[cluster_axis]
    hops = _hop_links(mesh_device, cluster_axis, ring)
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    L = num_links if num_links is not None else min(len(v) for v in hops.values())
    L = min(L, gx // 2)  # ports live in logical row 0, columns 0..2L-1 (auto only; explicit L is checked by the caller)

    shape = tuple(input_tensor.shape)
    H, W = shape[-2], shape[-1]
    batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
    tensor_row_tiles = batch * math.ceil(H / TILE)
    tensor_w_tiles = math.ceil(W / TILE)
    mem_config = input_tensor.memory_config()
    sharded = mem_config.memory_layout in SHARDED_LAYOUTS
    if sharded:
        lay = _sharded_layout(mem_config, tensor_row_tiles, tensor_w_tiles, gx, gy)
    else:
        lay = _interleaved_layout(tensor_row_tiles, tensor_w_tiles, gx, gy)
    S_r, S_w, core_rows = lay["S_r"], lay["S_w"], lay["core_rows"]

    L_eff = min(L, S_r)  # a link with no row group would carry no records (and no arrival increment)
    if lay["free_cores"] is None:
        port_cores = [ttnn.CoreCoord(2 * l + d, 0) for l in range(L_eff) for d in (0, 1)]
    else:
        L_eff = min(L_eff, len(lay["free_cores"]) // 2)
        if L_eff == 0:
            raise RuntimeError("distributed_rms_norm: the shard grid leaves fewer than 2 free cores for the ports")
        port_cores = lay["free_cores"][: 2 * L_eff]
    if lay["free_cores"] is None:
        # interleaved: contiguous runs of row groups per link, so each link's workers form a few rectangles and a
        # port's stats-ready multicast reaches only its own link's workers (PER_LINK_MCAST)
        link_of = [g * L_eff // S_r for g in range(S_r)] if PER_LINK_MCAST else [g % L_eff for g in range(S_r)]
    else:
        link_of = [g % L_eff for g in range(S_r)]
    link_mcast_rects = [lay["mcast_rects"]] * L_eff
    if lay["free_cores"] is None and PER_LINK_MCAST:
        link_mcast_rects = [
            _linear_rects([wk["core"] for wk in lay["workers"] if link_of[wk["g"]] == l], gx) for l in range(L_eff)
        ]

    # R5 device-local pre-combine (sharded WIDTH): W chunk w of a row group publishes its records into the inbox
    # of the combiner owning that tile row (rows split in lc_per runs over the chunks); combiners reduce the S_w
    # records and publish ONE record per tile row. rec_w = records per tile row on the exchange.
    local_combine = sharded and S_w > 1
    rec_w = 1 if local_combine else S_w
    lc_per = [math.ceil(r / S_w) for r in core_rows] if local_combine else [0] * S_r
    publishers = [math.ceil(core_rows[g] / lc_per[g]) if local_combine else S_w for g in range(S_r)]

    n_src_tiles = math.ceil(G * rec_w / TILE)
    core_rows_max = max(core_rows)
    xg_dtype = xg_dtype_of(fp32_dest_acc_en)
    rounds = (
        lay["rounds"] if sharded else _fit_rounds(lay, input_tensor.dtype, gamma_dtype, n_src_tiles, xg_dtype=xg_dtype)
    )
    block_rows = block_rows_knob(core_rows_max, rounds)
    num_blocks = max(math.ceil(r / block_rows) for r in core_rows)
    assert num_blocks == 1 or not local_combine, "R5 pre-combine runs one round"
    if G > 32:
        raise RuntimeError("distributed_rms_norm: the reader's per-port slot mask holds at most 32 group members")
    if SEM_STATS_READY_BASE + 2 * L_eff > NUM_SEMAPHORES:
        raise RuntimeError(f"distributed_rms_norm: {L_eff} links need more than {NUM_SEMAPHORES} semaphores")

    # Record slot layout, round-major: round b of link l occupies [b * round_recs[l], (b + 1) * round_recs[l]) and
    # holds, per row group g of the link, min(block_rows, core_rows[g]) tile rows x rec_w records from
    # link_rec_base[g]. One round (R1) is the dense whole-call layout. A group with fewer rows in its last round
    # leaves a gap that is sent but never read.
    rows_b = [min(block_rows, r) for r in core_rows]
    round_recs = [rec_w * sum(rows_b[g] for g in range(S_r) if link_of[g] == l) for l in range(L_eff)]
    link_rec_base = [rec_w * sum(rows_b[h] for h in range(g) if link_of[h] == link_of[g]) for g in range(S_r)]

    # local arrivals at a port of link l by the end of round b, on round b's rotating counter (cumulative over the
    # rounds that share it): every publisher with rows in round b increments once
    def pubs(l, b):
        return sum(publishers[g] for g in range(S_r) if link_of[g] == l and core_rows[g] > b * block_rows)

    publishers_round = [
        [sum(pubs(l, c) for c in range(b % LOCAL_ARRIVAL_SEMS, b + 1, LOCAL_ARRIVAL_SEMS)) for b in range(num_blocks)]
        for l in range(L_eff)
    ]
    slot_bytes = RECORD_BYTES * num_blocks * max(round_recs)

    return dict(
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        ring=ring,
        G=G,
        groups=_groups(mesh_shape, cluster_axis),
        hops=hops,
        gx=gx,
        gy=gy,
        L=L_eff,
        W=W,
        tensor_row_tiles=tensor_row_tiles,
        tensor_w_tiles=tensor_w_tiles,
        sharded=sharded,
        S_r=S_r,
        S_w=S_w,
        core_w_tiles=lay["core_w_tiles"],
        core_rows=core_rows,
        row_start=lay["row_start"],
        workers=lay["workers"],
        worker_cores=lay["worker_cores"],
        worker_set=lay["worker_set"],
        link_mcast_rects=link_mcast_rects,
        port_cores=port_cores,
        link_of=link_of,
        round_recs=round_recs,
        publishers_round=publishers_round,
        link_rec_base=link_rec_base,
        slot_bytes=slot_bytes,
        n_src_tiles=n_src_tiles,
        rec_w=rec_w,
        local_combine=local_combine,
        lc_per=lc_per,
        n_lc_tiles=math.ceil(S_w / TILE) if local_combine else 0,
        block_rows=block_rows,
        num_blocks=num_blocks,
        xg_dtype=xg_dtype,
    )


def _worker_footprint(plan, x_dtype, gamma_dtype, gamma_prescale, stat_rows=None, gamma_chunk=None, out_native=False):
    """Per-worker CB bytes (l1_ledger.md F_worker). cb_xg exists only on the gamma-prescale path. A sharded x / a
    native sharded output are shard-backed CBs: they cost no CB bytes (the shard is a buffer of its own)."""
    T_x = _tile_bytes(x_dtype)
    T_f = _tile_bytes(ttnn.float32)
    br, cw = plan["block_rows"], plan["core_w_tiles"]
    s = br if stat_rows is None else stat_rows
    gc = cw if gamma_chunk is None else min(gamma_chunk, cw)
    has_gamma = gamma_dtype is not None
    nf = blocks_in_flight(plan["num_blocks"])
    prescale = has_gamma and gamma_prescale
    f = 0 if plan["sharded"] else X_BLOCKS_IN_FLIGHT * x_blocks(plan["num_blocks"], prescale) * br * cw * T_x
    if has_gamma:
        f += gc * (1 if gc >= cw else 2) * _tile_bytes(gamma_dtype)
    if prescale:
        f += nf * br * cw * _tile_bytes(plan.get("xg_dtype", ttnn.float32))
    f += _tile_bytes(ttnn.bfloat16)
    f += s * T_f * (3 + plan["n_src_tiles"])
    if plan["local_combine"]:
        f += max(plan["lc_per"]) * T_f * (plan["n_lc_tiles"] + 1)
    if not out_native:
        f += OUT_DEPTH * out_batch_tiles_knob(cw) * T_x
    return f


def _choose_knobs(plan, x_dtype, gamma_dtype, out_native, budget):
    """(stat_rows, gamma_chunk, gamma_prescale) — the coarsest that fits `budget` CB bytes per worker.

    Interleaved (R1): whole block, whole gamma; prescale when it fits. Sharded (R9/R10): the x block is the shard,
    so what remains to fit are the per-row stats CBs (streamed in stat_rows batches; WIDTH holds every tile row)
    and gamma (streamed in gamma_chunk tiles; HEIGHT holds the whole W_local)."""
    br, cw = plan["block_rows"], plan["core_w_tiles"]
    has_gamma = gamma_dtype is not None

    def fits(prescale, s, gc):
        return _worker_footprint(plan, x_dtype, gamma_dtype, prescale, s, gc, out_native) <= budget

    if not plan["sharded"]:
        prescale = has_gamma and GAMMA_PRESCALE and fits(True, br, cw)
        return br, cw, prescale
    halvings = lambda n: sorted({max(1, n >> k) for k in range(n.bit_length() + 1)}, reverse=True)
    for prescale in [True, False] if has_gamma else [False]:
        for gc in halvings(cw) if not prescale else [cw]:
            for s in halvings(br):
                if fits(prescale, s, gc):
                    return s, gc, prescale
    return br, cw, False  # does not fit at any knob setting: the footprint check reports it


# ----------------------------------------------------------------------------------------------------------------
# Persistent per-mesh resources
# ----------------------------------------------------------------------------------------------------------------

_SEM_CACHE = {}


def _global_semaphores(mesh_device, port_cores):
    """(arrival, ready) global semaphores over the port cores; created once, synchronized once, cached."""
    key = (id(mesh_device), tuple(sorted((c.x, c.y) for c in port_cores)))
    if key not in _SEM_CACHE:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in port_cores])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2))
        # every chip's counters must be zero before any chip's first increment can arrive
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, sems)
    return _SEM_CACHE[key][1]


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


# ----------------------------------------------------------------------------------------------------------------
# MeshProgramDescriptor
# ----------------------------------------------------------------------------------------------------------------


def _same_shards(a, b):
    """True when two sharded memory configs place the same shard on the same core (zero-copy output possible)."""
    if a.memory_layout != b.memory_layout or a.shard_spec is None or b.shard_spec is None:
        return False
    sa, sb = a.shard_spec, b.shard_spec
    return (
        a.buffer_type == b.buffer_type
        and sa.grid == sb.grid
        and tuple(sa.shape) == tuple(sb.shape)
        and sa.orientation == sb.orientation
    )


def _l1_free_below_buffers(mesh_device):
    """Bytes between the worker-L1 allocator base (where static CBs start) and the lowest allocated L1 buffer.

    L1 buffers are allocated top-down in lockstep across banks, CBs bottom-up from the same base, so this is the
    CB room every core of the program has. It counts every live L1 buffer (shards, scratch, inbox, and any other
    tensor the caller still holds), read from the allocator's block table."""
    view = ttnn._ttnn.device.GetMemoryView(mesh_device, ttnn.BufferType.L1)
    used = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(used) if used else int(view.total_bytes_per_bank)


def _l1_sharded_tensor(mesh_device, cores_set, n_cores, words):
    """uint32 [n_cores, words] tensor, one row per core of `cores_set`: same L1 address on every core and chip."""
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores_set, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([n_cores, words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, mem
    )


def run_distributed_rms_norm(input_tensor, gamma, plan, *, epsilon, compute_kernel_config, memory_config):
    mesh_device = plan["mesh_device"]
    G, L, gx = plan["G"], plan["L"], plan["gx"]
    S_r, S_w = plan["S_r"], plan["S_w"]
    Wt = plan["tensor_w_tiles"]
    sharded, local_combine = plan["sharded"], plan["local_combine"]
    has_gamma = gamma is not None
    g_dtype = gamma.dtype if has_gamma else None
    x_dtype = input_tensor.dtype

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), x_dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config
    )
    # zero-copy output: compute packs straight into the output shard (no writer store)
    out_native = sharded and _same_shards(input_tensor.memory_config(), output.memory_config())

    # ---- cores ----
    port_cores = plan["port_cores"]
    port_of = {(l, d): port_cores[2 * l + d] for l in range(L) for d in (0, 1)}
    worker_cores = plan["worker_cores"]
    worker_set = plan["worker_set"]
    port_set = _crs(port_cores)
    all_set = worker_set.merge(port_set)
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    mcast = []  # per link: stats-ready multicast rectangles (NoC coords), only cores of this program
    for rects in plan["link_mcast_rects"]:
        mcast.append([])
        for r0c, r1c in rects:
            v0, v1 = virt(r0c), virt(r1c)
            mcast[-1].append((v0.x, v0.y, v1.x, v1.y, (r1c.x - r0c.x + 1) * (r1c.y - r0c.y + 1)))

    # ---- exchange scratch: G slots per port core, L1 height-sharded, same address on every chip ----
    slot_bytes = plan["slot_bytes"]
    scratch = _l1_sharded_tensor(mesh_device, port_set, len(port_cores), G * slot_bytes // 4)
    scratch_addr = int(scratch.buffer_address())
    # ---- R5 pre-combine inbox: per combiner, lc_per tile rows x S_w records ----
    lc_per_max = max(plan["lc_per"])
    inbox = None
    inbox_addr = 0
    if local_combine:
        inbox = _l1_sharded_tensor(mesh_device, worker_set, len(worker_cores), lc_per_max * S_w * RECORD_BYTES // 4)
        inbox_addr = int(inbox.buffer_address())
    arrival_sem, ready_sem = _global_semaphores(mesh_device, port_cores)
    arrival_addr = int(ttnn.get_global_semaphore_address(arrival_sem))
    ready_addr = int(ttnn.get_global_semaphore_address(ready_sem))
    packet_bytes = (ttnn.get_tt_fabric_max_payload_size_bytes() // RECORD_BYTES) * RECORD_BYTES

    # ---- knobs that must fit L1 ----
    if sharded:
        # the shards, scratch and inbox are already allocated: what is left below them is the CB room
        budget = min(WORKER_L1_BUDGET, _l1_free_below_buffers(mesh_device))
    else:
        budget = WORKER_L1_BUDGET
    stat_rows, gamma_chunk, gamma_prescale = _choose_knobs(plan, x_dtype, g_dtype, out_native, budget)
    footprint = _worker_footprint(plan, x_dtype, g_dtype, gamma_prescale, stat_rows, gamma_chunk, out_native)
    if footprint > budget:
        raise RuntimeError(f"distributed_rms_norm: per-worker working set {footprint} B exceeds {budget} B")
    if G * slot_bytes > SCRATCH_BUDGET:
        raise RuntimeError(f"distributed_rms_norm: exchange scratch {G * slot_bytes} B exceeds budget")

    # ---- formats / sizes ----
    block_rows = plan["block_rows"]
    num_blocks = plan["num_blocks"]
    n_in_flight = blocks_in_flight(num_blocks)
    n_src_tiles = plan["n_src_tiles"]
    n_lc_tiles = plan["n_lc_tiles"]

    def cb(index, dtype, pages, cores):
        return ttnn.CBDescriptor(
            total_size=pages * _tile_bytes(dtype),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=_tile_bytes(dtype))
            ],
        )

    cbs = [
        cb(CB_SCALER, ttnn.bfloat16, 1, worker_set),
        cb(CB_SQ_ACC, ttnn.float32, stat_rows, worker_set),
        cb(CB_PARTIAL, ttnn.float32, stat_rows, worker_set),
        cb(CB_STATS_COMBINE, ttnn.float32, n_src_tiles * stat_rows, worker_set),
        cb(CB_RSTD, ttnn.float32, stat_rows, worker_set),
    ]
    if local_combine:
        cbs += [
            cb(CB_LC, ttnn.float32, n_lc_tiles * lc_per_max, worker_set),
            cb(CB_LC_OUT, ttnn.float32, lc_per_max, worker_set),
        ]
    if sharded:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_RESIDENT, input_tensor))
    if out_native:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output))
    # CBs whose batch is a tile row of the W chunk are sized from the chunk the core actually owns: CB pointers
    # wrap only on an exact hit of the ring end, so every batch must divide the capacity. A ragged last chunk
    # (interleaved split, or the last WIDTH shard) gets its own CB descriptors. Idle cores join the first group.
    cw_of = {(wk["core"].x, wk["core"].y): wk["cw"] for wk in plan["workers"]}
    groups = {}
    for c in worker_cores:
        groups.setdefault(cw_of.get((c.x, c.y), plan["core_w_tiles"]), []).append(c)
    gamma_chunked = gamma_chunk < plan["core_w_tiles"]
    out_batch_of = {}
    for cw_group, cores in groups.items():
        cores_set = _crs(cores)
        gc_group = min(gamma_chunk, cw_group)
        # out batch must divide every normalize chunk (the writer walks them in out_batch steps)
        out_batch = out_batch_tiles_knob(cw_group) if (gc_group == cw_group or gamma_prescale) else 1
        assert cw_group % out_batch == 0, "out_batch_tiles must divide the core's W chunk"
        out_batch_of[cw_group] = out_batch
        if not sharded:
            n_x = X_BLOCKS_IN_FLIGHT * x_blocks(num_blocks, gamma_prescale)
            cbs.append(cb(CB_X_RESIDENT, x_dtype, n_x * block_rows * cw_group, cores_set))
        if not out_native:
            cbs.append(cb(CB_OUTPUT_TILES, x_dtype, OUT_DEPTH * out_batch, cores_set))
        if has_gamma:
            # one chunk held for the call, or a double-buffered stream of gamma_chunk-tile chunks
            cbs.append(cb(CB_GAMMA_TILES, gamma.dtype, gc_group * (1 if gc_group >= cw_group else 2), cores_set))
        if gamma_prescale:
            cbs.append(cb(CB_XG, plan["xg_dtype"], n_in_flight * block_rows * cw_group, cores_set))

    # ---- compute config (user knobs) + UnpackToDestFp32 tags ----
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=fp32_dest,
        math_approx_mode=bool(compute_kernel_config.math_approx_mode),
    )
    if fp32_dest and UNPACK_STATS_TO_DEST:
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
        for c in STATS_UNPACK_TO_DEST_CBS:
            modes[c] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = modes

    # ---- compile-time args ----
    x_acc = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    g_acc = list(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    o_acc = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    gamma_elem = (4 if gamma.dtype == ttnn.float32 else 2) if has_gamma else 2
    reader_ct = (
        [
            CB_X_RESIDENT,
            CB_GAMMA_TILES,
            CB_SCALER,
            CB_STATS_COMBINE,
            int(has_gamma),
            SEM_STATS_READY_BASE,
            gamma_elem,
            int(gamma.buffer_aligned_page_size()) if has_gamma else 0,
            RECORD_BYTES,
            int(has_gamma and not gamma_prescale),
            int(sharded),
            CB_LC,
            SEM_LOCAL_COMBINE,
            plan["W"] % TILE,  # w_tail: valid columns of the last W tile (0 = tile-aligned, no mask)
            ROUND_LOOKAHEAD,
            BFP8_EXP_BYTES if x_dtype == ttnn.bfloat8_b else 0,  # x_exp_bytes
            int(GAMMA_EARLY),
        ]
        + x_acc
        + g_acc
    )
    writer_ct = [
        CB_PARTIAL,
        CB_OUTPUT_TILES,
        SEM_LOCAL_ARRIVALS,
        RECORD_BYTES,
        int(out_native),
        int(local_combine),
        CB_LC_OUT,
        SEM_LOCAL_COMBINE,
        int(sharded),
        ROUND_LOOKAHEAD,
    ] + o_acc
    compute_ct = [
        CB_X_RESIDENT,
        CB_GAMMA_TILES,
        CB_SCALER,
        CB_SQ_ACC,
        CB_PARTIAL,
        CB_XG,
        CB_STATS_COMBINE,
        CB_RSTD,
        CB_OUTPUT_TILES,
        int(has_gamma),
        int(gamma_prescale),
        int(sharded),
        int(out_native),
        int(local_combine),
        CB_LC,
        CB_LC_OUT,
        ROUND_LOOKAHEAD,
    ]
    port_ct = [SEM_LOCAL_ARRIVALS, HEADER_RING, LOCAL_ARRIVAL_SEMS]

    inv_w_bits = _f32_bits(1.0 / (G * plan["W"]))
    eps_bits = _f32_bits(epsilon)
    x_addr = int(input_tensor.buffer_address())
    g_addr = int(gamma.buffer_address()) if has_gamma else 0
    o_addr = int(output.buffer_address())
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    # sharded-path runtime args are appended only on the sharded path: the interleaved per-core arg lists stay the
    # Phase 0 ones (each extra word is dispatched to ~110 cores on every device; the launch skew it adds between
    # devices shows up as exchange wait)
    sharded_args = (lambda args: list(args)) if sharded else (lambda args: [])

    workers = plan["workers"]
    by_group = {}
    for wk in workers:
        by_group.setdefault(wk["g"], []).append(wk)
    for g in by_group:
        by_group[g].sort(key=lambda wk: wk["w"])

    mesh_desc = ttnn.MeshProgramDescriptor()
    for grp in plan["groups"]:
        for p, coord in enumerate(grp):
            fwd, bwd = _schedule(p, G, plan["ring"])
            sends = {0: fwd, 1: bwd}
            nxt = grp[(p + 1) % G] if (plan["ring"] or p < G - 1) else None
            prv = grp[(p - 1) % G] if (plan["ring"] or p > 0) else None
            peer_of = {0: nxt, 1: prv}
            # what lands at my port (l, d): the upstream device's same-direction port sends it
            up_of = {0: prv, 1: nxt}
            up_pos = {0: (p - 1) % G, 1: (p + 1) % G}
            inbound = {}
            for d in (0, 1):
                inbound[d] = _schedule(up_pos[d], G, plan["ring"])[d] if up_of[d] is not None else []
            holds = {d: set(sends[d]) | set(inbound[d]) for d in (0, 1)}
            src_dir = [0 if q in holds[0] else 1 for q in range(G)]
            assert all(q in holds[src_dir[q]] for q in range(G)), "every slot must land at one of my ports"

            program = ttnn.ProgramDescriptor()
            program.cbs = cbs
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=sem, core_ranges=all_set, initial_value=0)
                for sem in [SEM_LOCAL_ARRIVALS + i for i in range(LOCAL_ARRIVAL_SEMS)]
                + [SEM_LOCAL_COMBINE]
                + [SEM_STATS_READY_BASE + i for i in range(2 * L)]
            ]

            # ---- worker runtime args ----
            # Common args: one copy per device (dispatch multicasts them); the per-link tables hold what differs by
            # link only. Per-core args: only what differs between cores. Every per-core word is dispatched to every
            # worker of every device, one device after the other, so per-core bytes are cross-device launch skew.
            pub_dirs = [d for d in (0, 1) if sends[d]]
            reader_common = [x_addr, g_addr, block_rows, Wt, scratch_addr, slot_bytes, plan["rec_w"], n_src_tiles, G]
            writer_common = [
                o_addr,
                block_rows,
                Wt,
                scratch_addr + p * slot_bytes,
                S_w if local_combine else plan["rec_w"],
                len(pub_dirs),
            ]
            for l in range(L):
                reader_common.append(plan["round_recs"][l])
                for q in range(G):
                    v = virt(port_of[(l, src_dir[q])])
                    reader_common += [v.x, v.y]
                reader_common.append(sum(1 << q for q in range(G) if src_dir[q] == 1))  # bwd_mask
                writer_common.append(plan["round_recs"][l])
                for d in pub_dirs:
                    v = virt(port_of[(l, d)])
                    writer_common += [v.x, v.y]
            compute_common = [block_rows, n_src_tiles, inv_w_bits, eps_bits]
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            used = set()
            for wk in workers:
                core, g, w = wk["core"], wk["g"], wk["w"]
                used.add((core.x, core.y))
                l = plan["link_of"][g]
                cw_here, cw_phys, rows = wk["cw"], wk["cw_phys"], wk["rows"]
                me = virt(core)
                lc_per = plan["lc_per"][g]
                lc_start = w * lc_per
                lc_rows = max(0, min(lc_per, rows - lc_start)) if local_combine else 0
                combiners = by_group[g][: math.ceil(rows / lc_per)] if local_combine else []
                reader_rt[core.x][core.y] = [
                    wk["row_start"],
                    rows,
                    wk["w_start"],
                    cw_here,
                    plan["link_rec_base"][g],
                    l,
                ] + sharded_args(
                    [
                        cw_phys,
                        stat_rows,
                        min(gamma_chunk, cw_here),
                        lc_rows,
                        n_lc_tiles,
                        len(by_group[g]),
                        inbox_addr,
                        me.x,
                        me.y,
                    ]
                )
                comb_args = []
                for cwk in combiners:
                    v = virt(cwk["core"])
                    comb_args += [v.x, v.y]
                norm_chunk = cw_here if (gamma_prescale or not has_gamma) else min(gamma_chunk, cw_here)
                writer_rt[core.x][core.y] = (
                    [
                        wk["row_start"],
                        rows,
                        wk["w_start"],
                        cw_here,
                        plan["link_rec_base"][g],
                        w,
                        out_batch_of[cw_here],
                        l,
                    ]
                    + sharded_args([stat_rows, norm_chunk, lc_rows, lc_start, inbox_addr, lc_per, len(combiners)])
                    + comb_args
                )
                compute_rt[core.x][core.y] = [rows, cw_here] + sharded_args(
                    [cw_phys, stat_rows, min(gamma_chunk, cw_here), lc_rows, n_lc_tiles]
                )
            for core in worker_cores:  # idle cores of the worker set: zero rows (they still receive stats-ready)
                if (core.x, core.y) not in used:
                    reader_rt[core.x][core.y] = [0, 0, 0, 0, 0, 0]
                    writer_rt[core.x][core.y] = [0, 0, 0, 0, 0, 0, 1, 0]
                    compute_rt[core.x][core.y] = [0, 0]

            # ---- port runtime args ----
            port_rt = ttnn.RuntimeArgs()
            for l in range(L):
                for d in (0, 1):
                    core = port_of[(l, d)]
                    peer = peer_of[d]
                    my_sends = sends[d] if peer is not None else []
                    # the peer's opposite-direction port writes into me: tell it I've started (ready fence)
                    send_ready = peer is not None and _peer_sends(p, d, G, plan["ring"])
                    has_conn = peer is not None and (bool(my_sends) or send_ready)
                    rv = virt(port_of[(l, 1 - d)])
                    pv = virt(port_of[(l, d)])
                    args = [
                        scratch_addr,
                        slot_bytes,
                        plan["round_recs"][l] * RECORD_BYTES,  # one entry = one round of the slot
                        packet_bytes,
                        arrival_addr,
                        ready_addr,
                        SEM_STATS_READY_BASE + 2 * l + d,
                        num_blocks,
                        len(inbound[d]),
                        int(send_ready),
                        rv.x,
                        rv.y,
                        int(has_conn),
                        pv.x,
                        pv.y,
                    ]
                    if peer is not None:
                        pn = node(peer)
                        args += [int(pn.mesh_id), int(pn.chip_id)]
                    else:
                        args += [0, 0]
                    args += [len(my_sends)] + list(my_sends)
                    # cumulative local arrivals per round (0: I send nothing, no local wait)
                    args += list(plan["publishers_round"][l]) if my_sends else [0] * num_blocks
                    args += [len(mcast[l])]
                    for rect in mcast[l]:
                        args += list(rect)
                    if has_conn:
                        link = plan["hops"][(coord, peer)][l]
                        args += list(ttnn.setup_fabric_connection(node(coord), node(peer), link, program, core))
                    port_rt[core.x][core.y] = args

            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    common_runtime_args=reader_common,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=writer_ct,
                    runtime_args=writer_rt,
                    common_runtime_args=writer_common,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=compute_ct,
                    runtime_args=compute_rt,
                    common_runtime_args=compute_common,
                    config=compute_config,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port.cpp"),
                    core_ranges=port_set,
                    compile_time_args=port_ct,
                    runtime_args=port_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=PORT_NOC
                    ),
                ),
            ]
            r, c = coord
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program

    tensors = [input_tensor] + ([gamma] if has_gamma else []) + [scratch] + ([inbox] if inbox is not None else [])
    tensors.append(output)
    return ttnn.generic_op(tensors, mesh_desc)


def _peer_sends(p, d, G, ring):
    """Whether the neighbour in direction d sends toward me (its opposite-direction port has entries)."""
    q = (p + 1) % G if d == 0 else (p - 1) % G
    fwd, bwd = _schedule(q, G, ring)
    return bool(bwd if d == 0 else fwd)
