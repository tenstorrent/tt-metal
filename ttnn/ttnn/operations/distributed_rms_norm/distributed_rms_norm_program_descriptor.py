# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm MeshProgramDescriptor (op_design.md -> Blocking Model, Dataflow Strategy, Work Distribution).

One program per chip, one ttnn.generic_op for the mesh. Per chip:

  port cores (2L, RISCV_0 / NoC1): the TRANSPORT. Ready fence, relay sends of the per-wave record slots over the
      ring / line, arrival counters, notifies the roots per wave, re-arm (kernels/distributed_rms_norm_port.cpp,
      adapted from examples/fabric_ring_exchange/kernels/port.cpp).
  compute cores (the grid left after the ports): the base op's local regimes, re-run on that grid:
      R1    interleaved, row-parallel, every core its own root (G_w = 1)
      R3-i  interleaved, W-split row-groups of G_w cores, the group's first core is its root
      R1-HS HEIGHT_SHARDED, every shard core its own root (zero-copy shard CBs)
      R3-s  WIDTH_SHARDED, one row-group over the shard cores (zero-copy shard CBs)
    Per block of R tile-rows, per wave of Rw rows: Σx² -> (R3: in-device combine at the root) -> device partial ->
    the root publishes one fp32 record (32 words) per tile-row into its link's port landing (slot p), the ports
    relay the slots, the root collects all G slots, sums them in slot order 0..G-1 in fp32 (bit-identical on every
    chip), finalizes, distributes inv_rms to its members; every core scales and writes the wave.

Knobs (single source of truth; every CB page count, trip count and the grid derive from these):
    W_CHUNK_TILES  cap on the streaming chunk C (halved until the blocking fits L1)
    STREAM_DEPTH   depth of cb_output and of the streaming cb_gamma_tiles, in C-tile quanta
    GAMMA_RESIDENT prefer the resident γ window (Wp pages, read once per call); when the blocking does not fit with it
                   (wide-row shards: e.g. a 512 KiB input + 512 KiB output shard on one core), γ streams per (block,
                   chunk) at STREAM_DEPTH * C pages (base op's L1 fallback). False forces streaming everywhere.
    SCALE_BLOCK_TILES  tiles per DEST window of the scaling chains (clamped by the chain to DEST capacity)
    MIN_BLOCKS_PER_CORE  cap R at ceil(rows_max / MIN_BLOCKS_PER_CORE). The base op uses 2 (prefetch of block k+1);
                   here 1: a block's waves already overlap its reads / exchange / scale, while splitting a core's rows
                   into blocks serializes one exchange round per block. Measured (ring 1x4, 2 links, fastest chip):
                   640x1792 30.9 -> 23.6 us (slowest 31.5 -> 25.5), other perf cases unchanged (1 row per core).
    NUM_WAVES      target waves per block (lamp L1); Rw = R / nwpb with nwpb the largest divisor of R <= NUM_WAVES
                   (every full block's waves are exactly Rw rows, so every CB quantum divides its capacity)
    R3_*           the base op's interleaved W-split layout rule, run on N_fused = grid - 2L cores
    block_rows (R), x_depth  largest R whose L1 footprint fits (choose_block_rows)
    num_links (L)  ports = 2L; root g -> link g mod L
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32
FACE_DIM = 16
TILE_HW = TILE_DIM * TILE_DIM

# ---- Knobs (single source of truth) ----
W_CHUNK_TILES = 32
STREAM_DEPTH = 2
SCALE_BLOCK_TILES = 8
MIN_BLOCKS_PER_CORE = 1
GAMMA_RESIDENT = True
NUM_WAVES = 2
# Lamp L4: the ports notify the roots per landed entry (True) or once per wave (False). The root collects slot by
# slot either way; per entry it reads each slot while the next one is still relaying.
NOTIFY_PER_ENTRY = True
R3_ROWS_PER_GROUP = 2
R3_MAX_GROUP_CORES = 24
R3_MIN_SLICE_TILES = 4
R3_OCCUPANCY_GAIN = 2
R3_MIN_GROUP_CORES = 4
# L1 reserved for the persistent port landing scratch when sizing the compute blocks (the landing is an L1 buffer
# allocated on the port cores, which the allocator reserves on every core).
LANDING_L1_RESERVE = 64 * 1024
# Mechanism cap (op_design.md -> Mechanism caps): the landing holds every (wave, slot) of the call with no reuse, so it
# grows with T. Above this many bytes per port the call is refused (NotImplementedError) instead of letting the
# allocator OOM; the landing L1 is reserved on every core's bank, so it also comes out of every compute core's budget.
PORT_LANDING_MAX_BYTES = 512 * 1024

# ---- Transport constants ----
RECORD_BYTES = TILE_DIM * 4  # one fp32 word per row of a tile-row
SEG_ALIGN = 64  # landing slot alignment (Blackhole DRAM 64 B covers L1 16 B)
MAX_HEADERS = 8  # packet headers per port (one per destination chip)
FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}

SCALER_FORMAT = ttnn.bfloat16

# ---- CB indices (semantic names; the kernels declare the same slots) ----
CB_X_TILES = 1
CB_REDUCE_SCALER = 2
CB_SUMSQ_PARTIAL = 3
CB_INV_RMS = 4
CB_GAMMA_TILES = 6
CB_X_GAMMA = 7
CB_PARTIAL_COL = 9
CB_GATHER = 10
CB_RMS_ROOT = 11
CB_SUMSQ_LAST = 12
CB_SUMSQ_ACC = 13
CB_OUTPUT = 16
CB_DEV_PARTIAL = 17
CB_STATS_GATHER = 18
CB_STATS_RECORD = 19

# ---- Program-local semaphores (declared on every compute and port core, before the fabric setup) ----
SEM_INV = 0  # members: the root's inv_rms writes (one ordered sender -> one cumulative counter)
SEM_OWN_STATS = 1  # ports: publications of the link's roots
SEM_LANDED_FWD = 2  # roots: entries landed on the link's forward port
SEM_LANDED_BWD = 3  # roots: entries landed on the link's backward port
SEM_OWN_COUNTED = 4  # roots: the link's sending ports counted wave w
# R3 roots: member partial arrivals, ONE PER WAVE OF A BLOCK (ids SEM_GATHER_BASE + k). G_w - 1 members send each
# wave: on one cumulative counter a fast member's wave-(k+1) increment could satisfy the wave-k threshold before a slow
# member's wave-k partial landed (measured: stale wave-0 rows at 640x1792 with 2 waves per block). Across blocks the
# per-wave count stays cumulative: a member sends block b+1 only after it received block b's inv_rms.
SEM_GATHER_BASE = 5
MAX_WAVES_PER_BLOCK = 8  # 5 + 8 program semaphores + 2 per fabric connection <= 16 ids per core
SEM_LANDED = {FWD: SEM_LANDED_FWD, BWD: SEM_LANDED_BWD}


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _align_up(v, a):
    return (v + a - 1) // a * a


# ---------------------------------------------------------------------------
# Groups, links, exchange schedule (TRANSPORT: relay, from examples/fabric_ring_exchange + fabric_all_gather)
# ---------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    """Ordered lists of mesh coords, one per group (position p = index)."""
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def usable_links(mesh, groups, ring):
    """min over every hop of every group (incl. the wrap hop of a ring) of the forwarding link count."""
    counts = []
    for grp in groups:
        G = len(grp)
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring:
            hops.append((grp[-1], grp[0]))
        for a, b in hops:
            counts.append(len(ttnn.get_forwarding_link_indices(_node(mesh, a), _node(mesh, b))))
            counts.append(len(ttnn.get_forwarding_link_indices(_node(mesh, b), _node(mesh, a))))
    return min(counts) if counts else 0


def exchange_plan(groups, ring):
    """Per chip coord: {p, next, prev, items[d], entries[d], ready_out[d]} — relay all-gather of one slot per chip.
    items[d]     slots port d sends, in order: item 0 = own slot p, item i >= 1 relays entry i-1 of port d
    entries[d]   slots landing on port d (sent by the upstream chip's port d), entry i has arrival counter i
    ready_out[d] chips whose port (the opposite direction) writes into this chip and must get a ready from port d"""
    chips = {}
    for grp in groups:
        G = len(grp)
        for p, coord in enumerate(grp):
            nxt = grp[(p + 1) % G] if (ring or p < G - 1) else None
            prv = grp[(p - 1) % G] if (ring or p > 0) else None
            if ring:
                kf = G // 2
                kb = G - 1 - kf
                fwd = [(p - i) % G for i in range(kf)]
                bwd = [(p + i) % G for i in range(kb)]
            else:
                fwd = [p - i for i in range(p + 1)] if p < G - 1 else []
                bwd = [p + i for i in range(G - p)] if p > 0 else []
            chips[coord] = dict(
                p=p,
                G=G,
                next=nxt,
                prev=prv,
                items={FWD: fwd, BWD: bwd},
                entries={FWD: [], BWD: []},
                ready_out={FWD: [], BWD: []},
            )
    for coord, ch in chips.items():
        for d, peer in ((FWD, ch["next"]), (BWD, ch["prev"])):
            if ch["items"][d]:
                assert peer is not None
                chips[peer]["entries"][d] = list(ch["items"][d])
                chips[peer]["ready_out"][_OPP[d]].append((coord, d))
    return chips


# ---------------------------------------------------------------------------
# Port placement and the fused grid
# ---------------------------------------------------------------------------


def _cores_of(core_range_set):
    cores = []
    for rng in core_range_set.ranges():
        for y in range(rng.start.y, rng.end.y + 1):
            for x in range(rng.start.x, rng.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


def _yx_sorted(cores):
    return sorted(cores, key=lambda c: (c.y, c.x))


def _crs(cores):
    """CoreRangeSet of the given cores merged into rectangles: maximal x-runs per row, then identical runs on
    consecutive rows stacked (DISPATCH-BOUND guard: no per-core CoreRange lists)."""
    runs = []
    run = None
    for c in _yx_sorted(cores):
        if run is not None and c.y == run[2] and c.x == run[1] + 1:
            run = (run[0], c.x, c.y)
        else:
            if run is not None:
                runs.append(run)
            run = (c.x, c.x, c.y)
    if run is not None:
        runs.append(run)
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in runs:
        for r in rects:
            if r[0] == x0 and r[1] == x1 and r[3] == y - 1:
                r[3] = y
                break
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in rects])


def place_ports(grid, num_links, busy):
    """2L port cores among the cores not in `busy` (shard grids): lowest logical y first (nearest the Ethernet row),
    then nearest the grid's x-centre. Interleaved: (x0 + i, 0), x0 = grid.x // 2 - L (the ring-exchange placement).
    Returns {(direction, link): core}."""
    n = 2 * num_links
    x0 = max(0, grid.x // 2 - num_links)
    cand = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in busy]
    cand.sort(key=lambda xy: (xy[1], abs(xy[0] - (x0 + (n - 1) / 2.0)), xy[0]))
    if len(cand) < n:
        raise NotImplementedError(
            f"distributed_rms_norm: the shard grid leaves {len(cand)} free cores, {n} port cores are needed"
        )
    chosen = sorted(cand[:n], key=lambda xy: (xy[1], xy[0]))
    cores = [ttnn.CoreCoord(x, y) for x, y in chosen]
    return {(d, l): cores[2 * l + d] for l in range(num_links) for d in (FWD, BWD)}


# ---------------------------------------------------------------------------
# Geometry, L1 footprint, blocking
# ---------------------------------------------------------------------------


def datum_bytes(dtype) -> int:
    return 1 if dtype == ttnn.bfloat8_b else ttnn.tile_size(dtype) // TILE_HW


@dataclass(frozen=True)
class Geometry:
    has_gamma: bool
    W: int
    Wt: int
    S: int  # tiles of the W-slice one core owns (nominal)
    C: int  # w_chunk_tiles
    num_w_chunks: int
    Wp: int  # num_w_chunks * C
    WS: int  # cb_x_tiles / output-shard row stride: Wp (interleaved) or S (shard-backed)
    T_in: int
    T_out: int
    T_g: int
    es: int
    es_g: int
    gamma_is_tile: bool
    interm_format: object
    T_i: int
    sharded: bool
    combine: bool
    G_w: int
    masked: bool
    G: int  # devices per group
    gamma_resident: bool  # cb_gamma_tiles holds the whole window (Wp) vs streams STREAM_DEPTH * C pages
    out_native: bool  # cb_output IS the output shard (same placement as the input shard) vs the writer path


def compute_geometry(
    input_tensor, gamma, chunk_cap, fp32_acc, S, sharded, combine, G_w, masked, G, gamma_resident=True, out_native=None
) -> Geometry:
    W = list(input_tensor.shape)[-1]
    Wt = math.ceil(W / TILE_DIM)
    if sharded:
        # The shard's row stride is S: chunks of C = the largest divisor of S <= cap, so every chunk is C wide and
        # every per-row CB quantum is C (it divides the CB capacities).
        C = max(d for d in range(1, min(chunk_cap, S) + 1) if S % d == 0)
        num_w_chunks = S // C
    else:
        num_w_chunks = math.ceil(S / chunk_cap)
        C = math.ceil(S / num_w_chunks)
    Wp = num_w_chunks * C
    WS = S if sharded else Wp
    interm = ttnn.float32 if fp32_acc else ttnn.bfloat16
    has_gamma = gamma is not None
    return Geometry(
        has_gamma=has_gamma,
        W=W,
        Wt=Wt,
        S=S,
        C=C,
        num_w_chunks=num_w_chunks,
        Wp=Wp,
        WS=WS,
        T_in=ttnn.tile_size(input_tensor.dtype),
        T_out=ttnn.tile_size(input_tensor.dtype),
        T_g=ttnn.tile_size(gamma.dtype) if has_gamma else 0,
        es=datum_bytes(input_tensor.dtype),
        es_g=datum_bytes(gamma.dtype) if has_gamma else 0,
        gamma_is_tile=has_gamma and gamma.layout == ttnn.TILE_LAYOUT,
        interm_format=interm,
        T_i=ttnn.tile_size(interm),
        sharded=sharded,
        combine=combine,
        G_w=G_w,
        masked=masked,
        G=G,
        gamma_resident=has_gamma and gamma_resident,
        out_native=sharded if out_native is None else (sharded and out_native),
    )


def wave_split(block_rows: int):
    """(Rw, nwpb): nwpb = the largest divisor of R that is <= min(NUM_WAVES, MAX_WAVES_PER_BLOCK), Rw = R / nwpb."""
    cap = max(1, min(block_rows, NUM_WAVES, MAX_WAVES_PER_BLOCK))
    nwpb = max(d for d in range(1, cap + 1) if block_rows % d == 0)
    return block_rows // nwpb, nwpb


def gamma_tiles_pages(g: Geometry) -> int:
    """cb_gamma_tiles capacity: the whole padded window (resident) or STREAM_DEPTH C-tile chunks (streaming)."""
    return g.Wp if g.gamma_resident else STREAM_DEPTH * g.C


def scaler_pages(g: Geometry) -> int:
    """cb_reduce_scaler pages: the 1.0 scaler, plus the partial-width scaler of the sharded W-pad mask."""
    return 2 if g.masked else 1


def cb_inventory(g: Geometry, R: int, x_depth: int, input_dtype, output_dtype, gamma_dtype):
    """THE single source of every allocator-owned compute-core CB (l1_ledger.md): [(index, format, page_size, pages)].
    The L1 solve (footprint) and the program's CB descriptors are both built from this list, so a knob change lands
    in exactly one place. Shard-backed CBs (sharded cb_x_tiles / cb_output) are not listed: they live in the caller's
    shard and cost no CB L1. Root-only CBs are declared on every core (uniform CB layout)."""
    Rw, nwpb = wave_split(R)
    T_i, fi = g.T_i, g.interm_format
    inv = [
        (CB_REDUCE_SCALER, SCALER_FORMAT, ttnn.tile_size(SCALER_FORMAT), scaler_pages(g)),
        (CB_SUMSQ_PARTIAL, fi, T_i, Rw),
        (CB_INV_RMS, fi, T_i, R),
        (CB_DEV_PARTIAL, fi, T_i, nwpb * Rw),
        (CB_STATS_GATHER, fi, T_i, g.G * Rw),
        (CB_STATS_RECORD, ttnn.uint32, RECORD_BYTES, (g.G + 1) * Rw),
    ]
    if not g.sharded:
        inv.append((CB_X_TILES, input_dtype, g.T_in, x_depth * R * g.WS))
    if not g.out_native:
        inv.append((CB_OUTPUT, output_dtype, g.T_out, STREAM_DEPTH * g.C))
    if g.has_gamma:
        inv.append((CB_GAMMA_TILES, gamma_dtype, g.T_g, gamma_tiles_pages(g)))  # resident window or stream ring
        inv.append((CB_X_GAMMA, fi, T_i, R * g.C))
    if g.combine:
        inv.append((CB_PARTIAL_COL, fi, T_i, R))
        inv.append((CB_GATHER, fi, T_i, nwpb * g.G_w * Rw))
        inv.append((CB_RMS_ROOT, fi, T_i, nwpb * Rw))
    if g.masked:
        inv.append((CB_SUMSQ_LAST, fi, T_i, Rw))
        inv.append((CB_SUMSQ_ACC, fi, T_i, Rw))
    return inv


def footprint(g: Geometry, R: int, x_depth: int) -> int:
    """Per-compute-core CB bytes (l1_ledger.md), summed from cb_inventory (formats do not change the bytes)."""
    return sum(page * pages for _, _, page, pages in cb_inventory(g, R, x_depth, None, None, None))


def _max_block_rows(g, x_depth, cap, budget):
    best = 0
    for r in range(1, cap + 1):  # footprint is not strictly monotone through wave_split; scan
        if footprint(g, r, x_depth) <= budget:
            best = r
    return best


def choose_block_rows(g: Geometry, rows_per_core_max: int, budget: int):
    """(R, x_depth): the coarsest R <= ceil(rows_max / MIN_BLOCKS_PER_CORE) that fits L1. x_depth = 2 (block k+1 read
    while block k exchanges and scales) only when the core walks more than one block; one block needs one buffer."""
    if g.sharded:
        r = _max_block_rows(g, 1, rows_per_core_max, budget)
        return (r, 1) if r >= 1 else None
    cap = max(1, math.ceil(rows_per_core_max / MIN_BLOCKS_PER_CORE))
    r1 = _max_block_rows(g, 1, cap, budget)
    if r1 >= rows_per_core_max:
        return r1, 1  # one block per core: nothing to prefetch
    r2 = _max_block_rows(g, 2, cap, budget)
    if r2 >= 1:
        return r2, 2
    if r1 >= 1:
        return r1, 1
    return None


def choose_blocking(
    input_tensor, gamma, rows_max, budget, fp32_acc, S, sharded, combine, G_w, masked, G, out_native=None
):
    """Resident γ first (the γ window read once per call); when no blocking fits with it, the same search with γ
    streamed per (block, chunk) — the base op's L1 fallback for wide-row shards."""
    choice = None
    if GAMMA_RESIDENT or gamma is None:
        choice = _search_blocking(
            input_tensor, gamma, rows_max, budget, fp32_acc, S, sharded, combine, G_w, masked, G, True, out_native
        )
    if choice is None and gamma is not None:
        choice = _search_blocking(
            input_tensor, gamma, rows_max, budget, fp32_acc, S, sharded, combine, G_w, masked, G, False, out_native
        )
    return choice


def _search_blocking(
    input_tensor,
    gamma,
    rows_max,
    budget,
    fp32_acc,
    S,
    sharded,
    combine,
    G_w,
    masked,
    G,
    gamma_resident=True,
    out_native=None,
):
    fallback = None
    cap = W_CHUNK_TILES
    while cap >= 1:
        g = compute_geometry(
            input_tensor, gamma, cap, fp32_acc, S, sharded, combine, G_w, masked, G, gamma_resident, out_native
        )
        choice = choose_block_rows(g, rows_max, budget)
        if choice is not None:
            if choice[1] == 2 or sharded or choice[0] >= rows_max:
                return g, choice[0], choice[1]
            if fallback is None:
                fallback = (g, choice[0], choice[1])
        cap //= 2
    return fallback


# ---------------------------------------------------------------------------
# Work plans (regime selection on the fused grid)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreWork:
    core: object  # logical CoreCoord
    start_tile_row: int
    rows: int
    w0: int
    V: int
    index: int  # position in the row-group (0 = root)
    root: object  # logical CoreCoord of the row-group root


@dataclass(frozen=True)
class Plan:
    regime: str
    work: tuple
    geometry: Geometry
    block_rows: int
    x_depth: int
    rows_max: int

    @property
    def roots(self):
        return [w for w in self.work if w.index == 0]


def tensor_tile_rows_of(input_tensor) -> int:
    shape = list(input_tensor.shape)
    return _prod(shape[:-2]) * math.ceil(shape[-2] / TILE_DIM)


def _group_work(cores, group_size, S, Wt, row_ranges):
    work = []
    for gi, (start, rows) in enumerate(row_ranges):
        group = cores[gi * group_size : (gi + 1) * group_size]
        for j, core in enumerate(group):
            w0 = j * S
            V = min(S, Wt - w0)
            if V <= 0 or rows <= 0:
                raise NotImplementedError("distributed_rms_norm: a core with no valid data is not supported")
            work.append(CoreWork(core, start, rows, w0, V, j, group[0]))
    return work


def _r1_work(available, Wt, T):
    n = min(T, len(available))
    base, extra = divmod(T, n)
    work, start = [], 0
    for i in range(n):
        rows = base + (1 if i < extra else 0)
        c = available[i]
        work.append(CoreWork(c, start, rows, 0, Wt, 0, c))
        start += rows
    assert start == T
    return work, base + (1 if extra else 0)


def _r3_interleaved_layout(Wt, T, num_cores):
    groups_target = math.ceil(T / R3_ROWS_PER_GROUP)
    G_w_target = max(1, min(R3_MAX_GROUP_CORES, num_cores // groups_target))
    S = max(R3_MIN_SLICE_TILES, math.ceil(Wt / G_w_target))
    S = min(S, Wt)
    G_w = math.ceil(Wt / S)
    G_h = max(1, min(T, num_cores // G_w))
    return S, G_w, G_h


def same_placement(out_mc, in_mc) -> bool:
    """Output memory config identical in placement to the input's (interleaved / interleaved, or the same shard)."""
    if out_mc.memory_layout != in_mc.memory_layout:
        return False
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return True
    a, b = out_mc.shard_spec, in_mc.shard_spec
    return (
        out_mc.buffer_type == in_mc.buffer_type
        and a is not None
        and b is not None
        and list(a.shape) == list(b.shape)
        and a.grid == b.grid
        and a.orientation == b.orientation
    )


def output_is_native(input_tensor, output_tensor) -> bool:
    """cb_output backed zero-copy on the output shard: only when the output shard IS the input shard's placement
    (every core packs its own rows / columns into its own L1). Any other output placement (interleaved, or a shard on a
    different grid / shape) goes through the writer's TensorAccessor: a genuinely non-local write."""
    in_mc = input_tensor.memory_config()
    return in_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED and same_placement(
        output_tensor.memory_config(), in_mc
    )


def _sharded_work(input_tensor, T, Wt):
    mc = input_tensor.memory_config()
    spec = mc.shard_spec
    if spec is None or mc.buffer_type != ttnn.BufferType.L1:
        raise NotImplementedError("distributed_rms_norm: sharded inputs must carry a 2-D L1 shard spec")
    height = mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR and not height:
        raise NotImplementedError("distributed_rms_norm: only ROW_MAJOR shard orientation is supported")
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    assert shard_h % TILE_DIM == 0 and shard_w % TILE_DIM == 0, "TILE shards are tile-multiples"
    Hsh = shard_h // TILE_DIM
    S = shard_w // TILE_DIM
    if height:
        if S < Wt:
            raise NotImplementedError("distributed_rms_norm: height shard spec does not cover the row width")
        n = math.ceil(T / Hsh)
        cores = ttnn.corerange_to_cores(spec.grid, n, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        if len(cores) < n:
            raise NotImplementedError("distributed_rms_norm: height shard spec does not cover the tensor")
        ranges = [(i * Hsh, min(Hsh, T - i * Hsh)) for i in range(n)]
        return _group_work(list(cores), 1, S, Wt, ranges), S, max(r[1] for r in ranges)
    if mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise NotImplementedError(f"distributed_rms_norm: memory_layout {mc.memory_layout} is not built")
    cores = _yx_sorted(_cores_of(spec.grid))
    n = math.ceil(Wt / S)
    if Hsh < T or n > len(cores):
        raise NotImplementedError("distributed_rms_norm: width shard spec does not cover the tensor")
    return _group_work(cores[:n], n, S, Wt, [(0, T)]), S, T


def select_plan(input_tensor, gamma, fp32_acc, available, G, budget, out_native=None) -> Plan:
    W = list(input_tensor.shape)[-1]
    Wt = math.ceil(W / TILE_DIM)
    T = tensor_tile_rows_of(input_tensor)
    ml = input_tensor.memory_config().memory_layout

    def finish(regime, work, S, rows_max, sharded):
        G_w = max(w.index for w in work) + 1
        combine = G_w > 1
        # Compute-side W-pad mask (partial reduce scaler on the last tile): wherever the W-pad columns cannot be zeroed in
        # place — a caller's shard (read-only), or a Bfp8_b tile (16 datums share an exponent: no element-wise zero).
        masked = (W % TILE_DIM != 0) and (sharded or input_tensor.dtype == ttnn.bfloat8_b)
        choice = choose_blocking(
            input_tensor, gamma, rows_max, budget, fp32_acc, S, sharded, combine, G_w, masked, G, out_native
        )
        if choice is None:
            return None
        g, R, x_depth = choice
        return Plan(regime, tuple(work), g, R, x_depth, rows_max)

    if ml in (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED):
        work, S, rows_max = _sharded_work(input_tensor, T, Wt)
        avail = {(c.x, c.y) for c in available}
        if any((w.core.x, w.core.y) not in avail for w in work):
            raise NotImplementedError("distributed_rms_norm: a shard core collides with a port core")
        regime = "R1-HS" if ml == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else "R3-s"
        plan = finish(regime, work, S, rows_max, True)
        if plan is None:
            raise NotImplementedError("distributed_rms_norm: the shard does not fit one resident tile-row in L1")
        return plan
    if ml != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise NotImplementedError(f"distributed_rms_norm: memory_layout {ml} is not built")

    r1_work, rows_g1 = _r1_work(available, Wt, T)
    r1 = finish("R1", r1_work, Wt, rows_g1, False)
    S, G_w, G_h = _r3_interleaved_layout(Wt, T, len(available))
    wants_r3 = r1 is None or (
        R3_OCCUPANCY_GAIN > 0 and G_w >= R3_MIN_GROUP_CORES and G_w * G_h >= R3_OCCUPANCY_GAIN * len(r1_work)
    )
    if wants_r3 and G_w > 1:
        base, extra = divmod(T, G_h)
        ranges, start = [], 0
        for gi in range(G_h):
            rows = base + (1 if gi < extra else 0)
            ranges.append((start, rows))
            start += rows
        work = _group_work(available[: G_w * G_h], G_w, S, Wt, ranges)
        r3 = finish("R3-i", work, S, base + (1 if extra else 0), False)
        if r3 is not None:
            return r3
    if r1 is None:
        raise NotImplementedError(f"distributed_rms_norm: W={W} does not fit a resident tile-row in L1")
    return r1


# ---------------------------------------------------------------------------
# Persistent exchange resources (cached per mesh / port set)
# ---------------------------------------------------------------------------

_SEM_CACHE = {}


def _landing_words_bucket(words):
    b = 4096
    while b < words:
        b *= 2
    return b


def landing_bytes_for(landing_words: int) -> int:
    """Bytes per port core the landing actually occupies (its allocation bucket)."""
    return _landing_words_bucket(landing_words) * 4


def _whole_grid(mesh):
    grid = mesh.compute_with_storage_grid_size()
    return grid, ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])


def get_semaphores(mesh, G):
    """(ready semaphore, [G-1 arrival semaphores]): the only PERSISTENT exchange state, one grow-only list per opened
    mesh laid over the whole compute grid (every port placement reuses it; an L1 buffer's address range is reserved in
    every bank anyway). Persistent because a peer increments the ready counter before this chip's program for the call
    has started. A one-time synchronize_device at creation so every chip's counters are 0 before any peer's first
    increment; every call re-arms them to 0. The Python handles are kept alive here (generic_op keeps only addresses).
    """
    mid = int(mesh.id())  # unique per opened mesh (a Python id() can be recycled after a close / reopen)
    _, crs = _whole_grid(mesh)
    sems = _SEM_CACHE.setdefault(mid, [])
    if len(sems) < G:
        while len(sems) < G:
            sems.append(ttnn.create_global_semaphore(mesh, crs, 0))
        ttnn.synchronize_device(mesh)
    return sems[0], sems[1:G]


def allocate_landing(mesh, landing_words):
    """The exchange landing of ONE call (an L1 HEIGHT_SHARDED uint32 tensor over the whole grid; ports use their own
    shard). Per call, like an op intermediate: the mesh allocator gives it the same address on every chip, the ready
    fence keeps every peer write after this chip's program for the call has started, and the roots consume every
    arrival before the program ends, so it can be freed as soon as the call is enqueued. A persistent landing instead
    would be (re)allocated mid-call below the caller's L1 tensors and stay pinned in the middle of every bank, shrinking
    the CB space of every later call (verifier: golden sharded cells refused late in a long run, passing fresh)."""
    grid, crs = _whole_grid(mesh)
    words = _landing_words_bucket(landing_words)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([grid.x * grid.y, words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc
    )


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def _cb(index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def l1_budget(device, tensors=(), addrs=()) -> int:
    """Bytes the static CB region may span: it starts at the allocator base and must end below the LOWEST L1 buffer.
    The largest free block alone over-states it when a freed hole sits above a live buffer (measured: a wide-row shard
    planned against a hole clashed with its own output shard at 524160 B after an earlier call freed its shards), so it
    is also bounded by the lowest address of every L1 tensor this call knows (input / output / γ / landing) and of the
    persistent exchange semaphores (`addrs`: created on a mesh's first call, they sit just below that call's shards
    and pin every later call's CB region under them — measured: 524160 B below a 512 KiB shard)."""
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    base = ttnn._ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1)
    for t in tensors:
        if t is not None and t.memory_config().buffer_type == ttnn.BufferType.L1:
            budget = min(budget, t.buffer_address() - base)
    for a in addrs:
        budget = min(budget, a - base)
    return budget


def _pack(v):
    return (v.x << 16) | v.y


def _port_defines():
    """Measurement-only defines (never set in production: unset env -> no define, the production hash is unchanged).
    DRMS_ABLATE_EXCHANGE=1: the ports skip the fabric (op_design.md -> Measurement hooks)."""
    out = []
    if os.environ.get("DRMS_ABLATE_EXCHANGE", "0") not in ("", "0"):
        out.append(("DRMS_ABLATE_EXCHANGE", "1"))
    return out + _zone_defines()


def _zone_defines():
    """Measurement-only: DRMS_ZONES=1 turns on the per-stage device zones of every kernel (DRMS_ZONE in the kernels)."""
    if os.environ.get("DRMS_ZONES", "0") not in ("", "0"):
        return [("DRMS_ZONES", "1")]
    return []


def _compute_config(cfg, fp32_acc):
    out = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        fp32_dest_acc_en=cfg.fp32_dest_acc_en,
        math_approx_mode=cfg.math_approx_mode,
    )
    out.dst_full_sync_en = cfg.dst_full_sync_en
    if fp32_acc:
        # Lossless fp32 copy of the gathered partial records into DEST (copy_tile only; no srcA/B access).
        n = len(cfg.unpack_to_dest_mode) or 64
        modes = list(cfg.unpack_to_dest_mode) or [ttnn.UnpackToDestMode.Default] * n
        modes[CB_STATS_GATHER] = ttnn.UnpackToDestMode.UnpackToDestFp32
        out.unpack_to_dest_mode = modes
    return out


def create_mesh_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma,
    epsilon,
    compute_kernel_config,
    cluster_axis,
    G,
    num_links,
    ring,
):
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    L = num_links
    grid = mesh.compute_with_storage_grid_size()

    # ---- ports + fused grid ----
    in_mc = input_tensor.memory_config()
    out_native = output_is_native(input_tensor, output_tensor)
    busy = set()
    for t in (input_tensor, output_tensor):
        mc = t.memory_config()
        if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED and mc.shard_spec is not None:
            busy |= {(c.x, c.y) for c in _cores_of(mc.shard_spec.grid)}
    ports = place_ports(grid, L, busy)
    port_set = {(c.x, c.y) for c in ports.values()}
    available = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in port_set]

    # ---- local plan + exchange layout (the landing comes out of every core's L1, so plan against it) ----
    def exchange_layout(plan):
        R = plan.block_rows
        Rw, nwpb = wave_split(R)
        num_blocks_max = max(math.ceil(w.rows / R) for w in plan.work)
        num_waves = num_blocks_max * nwpb  # waves of the call (every root publishes every wave)
        n_roots = len(plan.roots)
        n_roots_l = [sum(1 for gi in range(n_roots) if gi % L == l) for l in range(L)]
        seg_l = [n * Rw * RECORD_BYTES for n in n_roots_l]
        seg_stride = _align_up(max(max(seg_l), 4), SEG_ALIGN)
        return Rw, nwpb, num_waves, n_roots_l, seg_l, seg_stride, num_waves * G * seg_stride // 4

    l1_tensors = (input_tensor, output_tensor, gamma)
    # Persistent semaphores first: their L1 words bound the CB region of this call too (l1_budget).
    ready_sem, arrival_sems = get_semaphores(mesh, G)
    ready_addr = int(ttnn.get_global_semaphore_address(ready_sem))
    arrival_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in arrival_sems]
    sem_addrs = [ready_addr] + arrival_addrs
    reserve = LANDING_L1_RESERVE
    for _ in range(4):
        try:
            plan = select_plan(
                input_tensor,
                gamma,
                fp32_acc,
                available,
                G,
                l1_budget(mesh, l1_tensors, sem_addrs) - reserve,
                out_native,
            )
        except NotImplementedError as exc:
            if reserve == LANDING_L1_RESERVE:
                raise
            raise NotImplementedError(
                f"distributed_rms_norm: the local plan does not fit L1 next to a {reserve} B exchange landing "
                "(long sequence; a landing reused across waves is a deferred regime)"
            ) from exc
        Rw, nwpb, num_waves, n_roots_l, seg_l, seg_stride, landing_words = exchange_layout(plan)
        need = landing_bytes_for(landing_words)
        if need > PORT_LANDING_MAX_BYTES:
            raise NotImplementedError(
                f"distributed_rms_norm: the exchange landing needs {need} B per port core (> {PORT_LANDING_MAX_BYTES}); "
                "a landing reused across waves (credit protocol) is a deferred regime"
            )
        if need <= reserve:
            break
        reserve = need  # re-plan against the landing this plan really needs (a smaller R may add waves)
    else:
        raise NotImplementedError(
            "distributed_rms_norm: the local plan and the exchange landing do not fit L1 together"
        )
    g, R, x_depth = plan.geometry, plan.block_rows, plan.x_depth
    roots = plan.roots
    root_link = {(w.core.x, w.core.y): (gi % L, gi // L) for gi, w in enumerate(roots)}

    landing = allocate_landing(mesh, landing_words)
    landing_addr = int(landing.buffer_address())
    if footprint(g, R, x_depth) > l1_budget(mesh, l1_tensors + (landing,), sem_addrs):
        ttnn.deallocate(landing)
        raise NotImplementedError("distributed_rms_norm: the CBs do not fit next to the exchange landing in L1")

    groups = build_groups(mesh_shape, cluster_axis)
    chips = exchange_plan(groups, ring)
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())

    compute_cores = [w.core for w in plan.work]
    compute_crs = _crs(compute_cores)
    port_crs = _crs(list(ports.values()))
    all_crs = _crs(compute_cores + list(ports.values()))
    virt = lambda c: mesh.worker_core_from_logical_core(c)

    # ---- CBs (identical on every chip; the same inventory the L1 solve summed) ----
    T_i, C = g.T_i, g.C
    gamma_dtype = gamma.dtype if gamma is not None else None
    cbs = [
        _cb(idx, fmt, page, pages, compute_crs)
        for idx, fmt, page, pages in cb_inventory(g, R, x_depth, input_tensor.dtype, output_tensor.dtype, gamma_dtype)
    ]
    if g.sharded:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input_tensor))
    if g.out_native:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor))

    w_valid_last = g.W % TILE_DIM or TILE_DIM
    es_zero = 0 if input_tensor.dtype == ttnn.bfloat8_b else g.es

    # ---- compile-time args (identical on every chip) ----
    reader_ct = [
        R,
        C,
        g.num_w_chunks,
        g.Wt,
        g.WS,
        x_depth,
        int(g.sharded),
        int(g.combine),
        g.G_w,
        Rw,
        nwpb,
        G,
        g.T_in,
        T_i,
        w_valid_last,
        es_zero,
        int(g.masked),
        num_waves,
        int(fp32_acc),
    ]
    reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    compute_ct = [
        int(g.has_gamma),
        R,
        C,
        g.num_w_chunks,
        g.WS,
        int(g.out_native),  # compute packs straight into the output shard
        int(g.combine),
        int(g.masked),
        g.G_w,
        SCALE_BLOCK_TILES,
        Rw,
        nwpb,
        G,
        int(g.gamma_resident),
    ]
    writer_ct = [
        int(g.has_gamma),
        int(g.gamma_is_tile),
        R,
        C,
        g.num_w_chunks,
        g.Wt,
        g.WS,
        int(g.out_native),  # nothing to write: cb_output is the output shard
        g.T_out,
        g.T_g,
        g.es_g,
        g.Wp,
        int(g.gamma_resident),
    ]
    writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    writer_ct += (
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    port_ct = [G, num_waves, payload, MAX_HEADERS, SEM_OWN_STATS, SEM_OWN_COUNTED]

    inv_w_bits = _fp32_bits(1.0 / (G * g.W))
    eps_bits = _fp32_bits(epsilon)
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    # row-group members (virtual coords) per root
    members = {}
    for w in plan.work:
        members.setdefault((w.root.x, w.root.y), {})[w.index] = virt(w.core)

    compute_cfg = _compute_config(compute_kernel_config, fp32_acc)
    base_key = "|".join(
        str(v)
        for v in (
            "distributed_rms_norm",
            list(input_tensor.shape),
            input_tensor.dtype,
            in_mc,
            output_tensor.memory_config(),
            gamma.dtype if gamma is not None else None,
            gamma.layout if gamma is not None else None,
            g.gamma_resident,
            g.out_native,
            plan.regime,
            [(w.core.x, w.core.y, w.start_tile_row, w.rows, w.w0, w.V, w.index) for w in plan.work],
            R,
            x_depth,
            C,
            g.num_w_chunks,
            Rw,
            nwpb,
            G,
            L,
            ring,
            cluster_axis,
            sorted(port_set),
            NOTIFY_PER_ENTRY,
            fp32_acc,
            compute_kernel_config.math_fidelity,
            compute_kernel_config.math_approx_mode,
            payload,
            _port_defines(),
        )
    )

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        program = ttnn.ProgramDescriptor()
        key = f"{base_key}|{coord}|{p}"
        program.custom_program_hash = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1
        # Our semaphores first: the fabric setup takes the lowest free ids on the port cores.
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=s, core_ranges=all_crs, initial_value=0) for s in range(SEM_GATHER_BASE + nwpb)
        ]
        program.cbs = cbs

        send_dirs = [d for d in (FWD, BWD) if ch["items"][d]]

        # slot q -> the port where it lands on this chip (own slot: the first sending port of the link)
        def slot_port(q, l):
            if q == p:
                return ports[(send_dirs[0], l)]
            for d in (FWD, BWD):
                if q in ch["entries"][d]:
                    return ports[(d, l)]
            raise AssertionError(f"slot {q} lands nowhere on chip {coord}")

        reader_rt = ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        writer_rt = ttnn.RuntimeArgs()
        for w in plan.work:
            core = w.core
            rv = virt(w.root)
            is_root = w.index == 0
            if is_root:
                l, r_l = root_link[(core.x, core.y)]
            else:
                l, r_l = 0, 0
            sp = [_pack(virt(ports[(d, l)])) for d in send_dirs] + [0] * (2 - len(send_dirs))
            args = [
                in_addr,
                w.start_tile_row,
                w.rows,
                w.w0,
                w.V,
                int(is_root),
                w.index,
                rv.x,
                rv.y,
                landing_addr,
                seg_stride,
                r_l * Rw * RECORD_BYTES,
                len(send_dirs),
                sp[0],
                sp[1],
                len(ch["entries"][FWD]),
                len(ch["entries"][BWD]),
                p,
            ]
            args += [_pack(virt(slot_port(q, l))) for q in range(G)]
            args += list(ch["entries"][FWD]) + list(ch["entries"][BWD])  # slot of entry i per direction
            mem = members[(w.root.x, w.root.y)]
            args += [_pack(mem[i]) if (is_root and i in mem) else 0 for i in range(1, g.G_w)]
            reader_rt[core.x][core.y] = args
            masked_core = int(g.masked and w.w0 + w.V == g.Wt)
            compute_rt[core.x][core.y] = [w.rows, inv_w_bits, eps_bits, w.V, int(is_root), masked_core]
            writer_rt[core.x][core.y] = [out_addr, gamma_addr, w.start_tile_row, w.rows, w.w0, w.V]

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
                core_ranges=compute_crs,
                compile_time_args=reader_ct,
                defines=_zone_defines(),
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
                core_ranges=compute_crs,
                compile_time_args=writer_ct,
                defines=_zone_defines(),
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                core_ranges=compute_crs,
                compile_time_args=compute_ct,
                defines=_zone_defines(),
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]

        # ---- ports ----
        port_rt = ttnn.RuntimeArgs()
        peer_of = {FWD: ch["next"], BWD: ch["prev"]}
        for l in range(L):
            link_roots = [_pack(virt(w.core)) for gi, w in enumerate(roots) if gi % L == l]
            for d in (FWD, BWD):
                core = ports[(d, l)]
                items, entries, ready_out = ch["items"][d], ch["entries"][d], ch["ready_out"][d]
                peer = peer_of[d]
                has_conn = len(items) > 0 or len(ready_out) > 0
                pc = virt(core)
                dests = [peer] if has_conn else []
                n_roots = len(link_roots) if items else 0
                n_notify = len(link_roots) if entries else 0
                n_counted = len(link_roots) if (items and num_waves > 1) else 0
                args = [
                    landing_addr,
                    seg_stride,
                    ready_addr,
                    len(dests),
                    len(items),
                    len(entries),
                    n_roots,
                    len(ready_out),
                    1 if items else 0,  # readies to wait for: one per destination chip (relay: the neighbour)
                    pc.x,
                    pc.y,
                    n_notify,
                    int(has_conn),
                    n_counted,
                    1 if NOTIFY_PER_ENTRY else max(1, len(entries)),  # notify quantum (lamp L4)
                    SEM_LANDED[d],
                ]
                args += arrival_addrs[: len(entries)]
                for dst in dests:
                    dn = _node(mesh, dst)
                    args += [int(dn.chip_id), int(dn.mesh_id), 1]
                for i, q in enumerate(items):
                    args += [q, 0, arrival_addrs[i], 0 if i == 0 else i]
                for tgt, td in ready_out:
                    assert tgt == peer
                    rc = virt(ports[(td, l)])
                    args += [0, rc.x, rc.y]
                args += [seg_l[l]] * num_waves
                args += link_roots if max(n_roots, n_notify) > 0 else []
                if has_conn:
                    pn = _node(mesh, peer)
                    link_ids = ttnn.get_forwarding_link_indices(_node(mesh, coord), pn)
                    args += list(ttnn.setup_fabric_connection(_node(mesh, coord), pn, link_ids[l], program, core))
                port_rt[core.x][core.y] = args
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port.cpp"),
                core_ranges=port_crs,
                compile_time_args=port_ct,
                defines=_port_defines(),
                runtime_args=port_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                ),
            )
        )
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc, landing
