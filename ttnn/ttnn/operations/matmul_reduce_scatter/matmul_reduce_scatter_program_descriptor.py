# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter — plan (blocking, schedule, placement) and the MeshProgramDescriptor.

One generic_op dispatch per call. Per chip:
  * compute rectangle (every grid row but the transport row(s)): a 2D-multicast matmul that walks the G scatter
    blocks in `compute_order`; each block is computed by the whole rectangle (per core `core_m_tiles x core_n_tiles`,
    K-blocked) and parked in the core's `cb_partial_handoff` slot (L1, backed by the `handoff_l1` sharded tensor).
      NCRISC: A operand (m-line injector + Mcast1D receivers) + hand-off ready/ack bookkeeping
      BRISC:  W operand (n-line injector + Mcast1D receivers)
      TRISC:  matmul_block (packer-L1 K accumulation, TileRowMajor hand-off)
  * transport row: per (direction, link) one port core, per (link, half) one final core — fabric_reduce_scatter's
    line transport, with "my partial" gathered from compute-core L1 instead of DRAM.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
BF16_TILE_BYTES = 2048

# ---- Blocking knobs (single source of truth; every dependent quantity derives from these) --------------------------
OPERAND_DEPTH = 2  # streamed operand K-blocks in flight (double buffering)
# hand-off slack per compute core, in scatter blocks (slots = handoff_depth * waves). The planner raises the depth from
# this floor toward G (one slot per scatter block: the matmul never waits on slot reuse, i.e. on a relay entry held by
# upstream arrivals -- Perf 2: FOCUS 166 -> 149 us, MiMo 341 -> 326 us) as far as L1 allows WITHOUT changing the
# floor-depth plan (regime, K-block, core block, waves): trading K-block for depth measured +15 us (K3 FFN, fp32 DEST).
HANDOFF_DEPTH_MIN = 2
BLOCKS_IN_FLIGHT = 1  # scatter blocks per compute pass (lamp L3; >1 is a future knob-turn)
STREAM_BUDGET = 384 * 1024  # bytes of streamed operand K-blocks per compute core
K_MIN_RESIDENT = 4  # residency must not force a degenerate K-block
# pipeline fill at K-block granularity (lamp L2): the first block's matmul starts after its first K-block (and, in R1,
# the resident operand is loaded K-block by K-block alongside it), so every compute pass has at least K_BLOCKS_MIN
# K-blocks (k_block_tiles <= Kt / K_BLOCKS_MIN, never below K_MIN_RESIDENT): a K-block of the whole K made the first
# block wait for the entire resident slice (MiMo: ~115 us of fill). MMRS_K_BLOCKS_MIN overrides (1 = uncapped).
K_BLOCKS_MIN = int(os.environ.get("MMRS_K_BLOCKS_MIN", "4"))
CORE_BLOCK_MAX = 64  # per-core block tiles (core_m_tiles * core_n_tiles)
# R4 sub-block sends: a scatter block is computed (and handed to the transport) as `waves` waves, each its own K pass
# on the whole grid, so the transport starts on wave 0 after ~1/waves of a block. Waves split the block along the
# scatter axis (rows for scatter_dim=-2, columns for -1): every wave then needs a disjoint slice of the streamed
# operand and the same slice of the block-invariant one, so nothing is re-streamed (a split across the other axis
# would make every wave re-read the block's whole streamed slice -- measured: the first wave stays stream-bound).
# The planner picks the largest waves <= WAVES_MAX that divides the scattered extent into whole segments and keeps the
# per-core work of a block within (1 + WAVES_WORK_SLACK) of the unwaved one. MMRS_WAVES pins a value (1 = unwaved).
WAVES_MAX = 1
WAVES_WORK_SLACK = 0.0
# line injectors (A and W): issue the next K-block's DRAM reads before multicasting the current one (overlaps the
# read with the multicast; 0 = the serial read -> barrier -> multicast walk). MMRS_READ_AHEAD overrides (A/B).
INJECT_READ_AHEAD = int(os.environ.get("MMRS_READ_AHEAD", "0"))
WAVES_PIN = int(os.environ["MMRS_WAVES"]) if os.environ.get("MMRS_WAVES") else None
L1_RESERVE = 64 * 1024  # L1 kept free on compute cores beyond the CBs and the hand-off shard (semaphores, misc)
XPORT_CB_BYTES = 112 * 1024  # transport CB sizing (reference)
XPORT_GROUP_MAX = 8
# transport placement (lamp L5): "eth" puts each port core next to its (direction, link) Ethernet core, per chip;
# "simple" is the fixed chip-independent layout (first 4L cores of the transport row)
XPORT_PLACEMENT = os.environ.get("MMRS_XPORT_PLACEMENT", "eth")
# Perf observability (never on in production): MMRS_PERF_ZONES=1 compiles the kernels' permanent MaybeDeviceZoneScope
# stage zones in (ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp; they need a --profile run too). MMRS_ABLATE is a
# comma list of payload ablations (MATMUL, OPERANDS, XREADS, LINK; results are WRONG): each stubs one stage's payload and
# keeps its synchronization, for /perf-measure's cumulative peeling.
_PERF_DEFINES = ([("KERNEL_PERF_ZONES", "1")] if os.environ.get("MMRS_PERF_ZONES") else []) + [
    (f"MMRS_ABLATE_{x.strip().upper()}", "1") for x in os.environ.get("MMRS_ABLATE", "").split(",") if x.strip()
]
# final cores per link (each owns 1/(F L) of the chip's own block: reads own partial + both arrivals, adds, stores);
# the transport row holds (2 + F) L cores
# W line multicast with rotating senders (every core of an n-line reads + multicasts every span-th K-block, rounds
# offset by the line index so one round's senders form a diagonal); 0 = one fixed injector per line
# relay ports: the port's BRISC (sender) reads the arrival segments from DRAM scratch (NoC1), its NCRISC only gathers the
# own partial -- a relay reader otherwise pulls two 14 KB streams per packet on one RISC
# transport cores' NoCs: 0 = readers (gathers, arrival reads, ack multicast) on NoC0, senders / final writers on NoC1;
# 1 = swapped (the transport row is the top grid row: NoC1 brings gathers from the compute rows straight up)
XPORT_NOC_SWAP = int(os.environ.get("MMRS_XPORT_NOC_SWAP", "0"))
PORT_ARR = int(os.environ.get("MMRS_PORT_ARR", "0"))
W_ROT = int(os.environ.get("MMRS_W_ROT", "0"))
FINALS_PER_LINK = int(os.environ.get("MMRS_FINALS_PER_LINK", "3"))
INC_EVERY = 8  # arrival-counter increment cadence (blackhole-fabric rule 4)
DEST_TILES_16B = 8  # DEST capacity in 16-bit tiles (half-sync); a 32-bit DEST (fp32_dest_acc_en) holds half
XPORT_ADD_BLOCK_MAX = DEST_TILES_16B // 2  # transport add: tiles per CB handshake / DEST batch (always fp32 DEST)

NOC0 = ttnn.NOC.NOC_0
NOC1 = ttnn.NOC.NOC_1

# ---- CB indices -------------------------------------------------------------------------------------------------
CB_ACT_OPERAND = 0
CB_WEIGHT_OPERAND = 1
CB_PARTIAL_ACCUM = 2
CB_PARTIAL_HANDOFF = 3
CB_XPORT_PARTIAL = 4
CB_XPORT_ARRIVAL_A = 5
CB_XPORT_ARRIVAL_B = 6
CB_XPORT_ZERO = 7  # one zero tile: the "+ 0" of the three-input final add
CB_XPORT_SUM = 16


def _cdiv(a, b):
    return -(-a // b)


def _divisors_desc(n):
    return [d for d in range(n, 0, -1) if n % d == 0]


def _tile_bytes(dtype):
    return (
        int(ttnn.tile_size(dtype))
        if hasattr(ttnn, "tile_size")
        else {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dtype]
    )


# ======================================================================================================================
# Blocking
# ======================================================================================================================


@dataclass
class Blocking:
    Mt: int
    Kt: int
    Nt: int
    G: int
    scatter_dim: int
    blk_m_tiles: int
    blk_n_tiles: int
    waves: int  # waves per scatter block (R4), along the scatter axis; one compute unit = one wave
    unit_m_tiles: int  # rows of one wave (blk_m_tiles / waves for scatter_dim=-2, else blk_m_tiles)
    unit_n_tiles: int  # columns of one wave (blk_n_tiles / waves for scatter_dim=-1, else blk_n_tiles)
    orientation: str  # "A": m-lines = grid rows; "B": m-lines = grid columns
    core_m_tiles: int
    core_n_tiles: int
    m_lines: int
    n_lines: int
    k_block_tiles: int
    num_k_blocks: int
    regime: str  # "R1" (invariant operand resident) or "R2" (both streamed)
    a_resident: bool
    w_resident: bool
    out_subblock_h: int
    out_subblock_w: int
    a_tile_bytes: int
    w_tile_bytes: int
    acc_dtype: object  # cb_partial_accum page format: follows the DEST width (fp32_dest_acc_en)
    acc_tile_bytes: int
    handoff_depth: int = HANDOFF_DEPTH_MIN  # hand-off slack in scatter blocks (planner: up to G when L1 allows)

    @property
    def handoff_slots(self):
        """cb_partial_handoff slots (one per compute unit): handoff_depth blocks of slack whatever the wave count, so
        splitting a block into waves never shrinks how far the matmul may run ahead of a stalled transport."""
        return self.handoff_depth * self.waves


def _factorize(unit_m, unit_n, comp_rows, comp_cols):
    """Grid factorization of one compute pass (unit_m x unit_n tiles) -> (orientation, cm, cn, m_lines, n_lines)."""
    best = None
    for orient, (m_avail, n_avail) in (("A", (comp_rows, comp_cols)), ("B", (comp_cols, comp_rows))):
        cm, cn = _cdiv(unit_m, m_avail), _cdiv(unit_n, n_avail)
        ml, nl = _cdiv(unit_m, cm), _cdiv(unit_n, cn)
        key = (cm * cn, -min(cm, cn), ml * nl)
        if best is None or key < best[0]:
            best = (key, orient, cm, cn, ml, nl)
    return best[1:]


def _seg_tiles(blk_n_tiles):
    """Tiles per transport segment (one fabric packet): the live payload, capped at a block row."""
    return max(1, min(int(ttnn.get_tt_fabric_max_payload_size_bytes()) // BF16_TILE_BYTES, blk_n_tiles))


def _plan_blocking(*, seg_tiles=None, **kw):
    """Blocking with the R4 wave count: the largest waves <= WAVES_MAX (or MMRS_WAVES) that splits the scattered
    extent of a block evenly (for scatter_dim=-1 into whole segments, so no segment straddles two waves) and keeps the
    per-core work of a block within WAVES_WORK_SLACK of the unwaved plan; else the unwaved plan."""
    Mt, Nt, G, sd = kw["Mt"], kw["Nt"], kw["G"], kw["scatter_dim"]
    blk_m, blk_n = (Mt // G, Nt) if sd == -2 else (Mt, Nt // G)
    seg = seg_tiles if seg_tiles is not None else _seg_tiles(blk_n)
    base = _plan_blocking_waves(waves=1, **kw)
    for waves in [WAVES_PIN] if WAVES_PIN else range(WAVES_MAX, 1, -1):
        extent = blk_m if sd == -2 else blk_n
        if waves <= 1 or extent % waves or (sd == -1 and (blk_n // waves) % seg):
            continue
        try:
            blk = _plan_blocking_waves(waves=waves, **kw)
        except ValueError:
            continue
        work = waves * blk.core_m_tiles * blk.core_n_tiles
        if WAVES_PIN or work <= (1 + WAVES_WORK_SLACK) * base.core_m_tiles * base.core_n_tiles:
            return blk
    return base


def _plan_blocking_waves(
    *, comp_rows, comp_cols, Mt, Kt, Nt, G, scatter_dim, a_dtype, w_dtype, fp32_acc, l1_cb_budget, waves
):
    """Grid factorization of one wave (unit_m x unit_n), K-block and regime (R1 resident invariant operand / R2
    streamed) per the design."""
    blk_m, blk_n = (Mt // G, Nt) if scatter_dim == -2 else (Mt, Nt // G)
    unit_m, unit_n = (blk_m // waves, blk_n) if scatter_dim == -2 else (blk_m, blk_n // waves)
    orient, cm, cn, ml, nl = _factorize(unit_m, unit_n, comp_rows, comp_cols)
    if cm * cn > CORE_BLOCK_MAX:
        raise ValueError(
            f"matmul_reduce_scatter: per-core block {cm}x{cn} tiles exceeds {CORE_BLOCK_MAX} "
            "(R4 waves split the scatter axis only; a per-core block this large needs a finer grid or a K split)"
        )

    a_tile, w_tile = _tile_bytes(a_dtype), _tile_bytes(w_dtype)
    acc_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16
    acc_tile = _tile_bytes(acc_dtype)
    accum = cm * cn * acc_tile
    budget = l1_cb_budget
    # R1: the block-invariant operand X resident for all G blocks (and all their waves: waves split the scatter axis,
    # along which X does not vary), the other (Y) streamed, a disjoint slice per wave.
    x_is_a = scatter_dim == -1
    core_x, tile_x = (cm, a_tile) if x_is_a else (cn, w_tile)
    core_y, tile_y = (cn, w_tile) if x_is_a else (cm, a_tile)
    resident = core_x * Kt * tile_x
    k_r1 = None
    for k in _divisors_desc(Kt):
        if k > max(Kt // K_BLOCKS_MIN, min(K_MIN_RESIDENT, Kt)):
            continue
        if k < min(K_MIN_RESIDENT, Kt):
            break
        if OPERAND_DEPTH * k * core_y * tile_y <= min(STREAM_BUDGET, budget - resident - accum):
            k_r1 = k
            break
    if k_r1 is not None:
        regime, kbt = "R1", k_r1
        a_res, w_res = x_is_a, not x_is_a
    else:
        regime, kbt = "R2", None
        for k in _divisors_desc(Kt):
            if k > max(Kt // K_BLOCKS_MIN, 1):
                continue
            if OPERAND_DEPTH * k * (cm * a_tile + cn * w_tile) <= min(STREAM_BUDGET, budget - accum):
                kbt = k
                break
        if kbt is None:
            raise ValueError("matmul_reduce_scatter: per-core block does not fit L1 even with one-tile K-blocks")
        a_res = w_res = False

    dest_limit = DEST_TILES_16B // 2 if fp32_acc else DEST_TILES_16B
    sb_w = next(d for d in _divisors_desc(cn) if d <= dest_limit)
    sb_h = next(d for d in _divisors_desc(cm) if d * sb_w <= dest_limit)
    return Blocking(
        Mt=Mt,
        Kt=Kt,
        Nt=Nt,
        G=G,
        scatter_dim=scatter_dim,
        blk_m_tiles=blk_m,
        blk_n_tiles=blk_n,
        waves=waves,
        unit_m_tiles=unit_m,
        unit_n_tiles=unit_n,
        orientation=orient,
        core_m_tiles=cm,
        core_n_tiles=cn,
        m_lines=ml,
        n_lines=nl,
        k_block_tiles=kbt,
        num_k_blocks=Kt // kbt,
        regime=regime,
        a_resident=a_res,
        w_resident=w_res,
        out_subblock_h=sb_h,
        out_subblock_w=sb_w,
        a_tile_bytes=a_tile,
        w_tile_bytes=w_tile,
        acc_dtype=acc_dtype,
        acc_tile_bytes=acc_tile,
    )


@dataclass
class Transport:
    seg_tiles: int
    seg_bytes: int
    segs_per_row: int
    segs_per_block: int
    xport_group: int
    cap_segs: int


def _plan_transport(blk: Blocking):
    seg_tiles = _seg_tiles(blk.blk_n_tiles)
    seg_bytes = seg_tiles * BF16_TILE_BYTES
    segs_per_row = _cdiv(blk.blk_n_tiles, seg_tiles)
    group = max(1, min(XPORT_GROUP_MAX, XPORT_CB_BYTES // (2 * seg_bytes)))
    return Transport(
        seg_tiles=seg_tiles,
        seg_bytes=seg_bytes,
        segs_per_row=segs_per_row,
        segs_per_block=blk.blk_m_tiles * segs_per_row,
        xport_group=group,
        cap_segs=2 * group,
    )


# ======================================================================================================================
# Schedule
# ======================================================================================================================


def _schedule_mmrs(p, G, ring=False):
    """Per-port block lists (send order), per-entry `has_upstream` flags and compute order for group position p.

    fwd: blocks sent toward p+1, farthest first; bwd: toward p-1. The last entry of each list is the downstream
    chip's own block. Linear: fwd `G-1 .. p+1`, bwd `0 .. p-1`; every entry has an upstream iff the chip has a
    neighbour behind it in that direction. Ring (G >= 3, design R3): each block's reduction chain is a line centred on
    its owner -- fwd `p+hf .. p+1`, bwd `p-hb .. p-1` (mod G), hf = ceil((G-1)/2), hb = G-1-hf; the first entry of each
    list (the farthest block of that direction) has no upstream (the chip's own partial starts the chain), the rest are
    relays. compute_order interleaves fwd/bwd one-by-one starting with fwd and ends with the own block p (the finals
    need it last). Upstream-less entries always precede relay entries (the transport add kernel copies them first)."""
    if ring and G >= 3:
        hf = (G - 1 + 1) // 2
        hb = G - 1 - hf
        fwd = [(p + d) % G for d in range(hf, 0, -1)]
        bwd = [(p - d) % G for d in range(hb, 0, -1)]
        fwd_up = [i > 0 for i in range(len(fwd))]
        bwd_up = [i > 0 for i in range(len(bwd))]
    else:  # Linear (a 2-device ring is the line)
        fwd = list(range(G - 1, p, -1))
        bwd = list(range(0, p))
        fwd_up = [p > 0] * len(fwd)
        bwd_up = [p < G - 1] * len(bwd)
    order = []
    for i in range(max(len(fwd), len(bwd))):
        if i < len(fwd):
            order.append(fwd[i])
        if i < len(bwd):
            order.append(bwd[i])
    order.append(p)
    return fwd, bwd, order, fwd_up, bwd_up


def _count_segs(first, stride, total):
    return _cdiv(total - first, stride) if first < total else 0


@dataclass
class Window:
    """A wave's place in its scatter block: tile origin (r0, c0) and its segments -- block rows [r0, r1) x segment
    columns [s0, s1) of each row (row-waves: whole rows, a contiguous segment range; column-waves: a column range of
    every row). Segments never straddle two waves (the planner splits columns into whole segments)."""

    r0: int
    c0: int
    r1: int
    s0: int
    s1: int

    def segs(self, first, stride, segs_per_row):
        """This core's segments (seg = first mod stride) in the window, in walk order: (first seg, count)."""
        mine = [
            r * segs_per_row + c
            for r in range(self.r0, self.r1)
            for c in range(self.s0, self.s1)
            if (r * segs_per_row + c) % stride == first % stride
        ]
        return (mine[0] if mine else 0), len(mine)


def _incs(n):
    return _cdiv(n, INC_EVERY) if n else 0


# ======================================================================================================================
# Placement
# ======================================================================================================================


@dataclass
class Placement:
    """Transport-row placement, per chip (lamp L5: each port sits next to the Ethernet core of its (direction, link),
    so it depends on the chip's fabric channels and harvesting). Port / final lists are keyed by mesh coord."""

    grid_x: int
    grid_y: int
    transport_rows: int
    finals_per_link: int = 2  # F: final cores per link (FINALS_PER_LINK, lowered when it would cost a compute row)
    mode: str = "simple"  # "eth" (ports under their Ethernet cores) or "simple" (first 4L cores of the row)
    fwd_ports: dict = field(default_factory=dict)  # coord -> [logical CoreCoord per link]
    bwd_ports: dict = field(default_factory=dict)
    finals: dict = field(default_factory=dict)  # coord -> [link 0 half 0, link 0 half 1, link 1 half 0, ...]
    vx: list = field(default_factory=list)  # virtual NoC x per logical column
    vy: list = field(default_factory=list)  # virtual NoC y per logical row


def _eth_channel(mesh_device, coord, peer, link):
    """Ethernet channel of the fabric router a worker on `coord` uses toward `peer` on `link`: the first word of the
    connection's runtime args (host-only -- the throwaway descriptor only collects the connection's semaphores)."""
    node = lambda c: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*c))
    return int(
        ttnn.setup_fabric_connection(node(coord), node(peer), link, ttnn.ProgramDescriptor(), ttnn.CoreCoord(0, 0))[0]
    )


def _eth_placer(mesh_device):
    """(coord, channel, taken, allowed) -> logical worker core with the fewest NoC1 hops (the sender's NoC) to that
    channel's Ethernet core; None when the chip's Ethernet layout is unknown (no cluster descriptor entry, emulator).
    Host-only: the physical Ethernet / worker-column maps come from the cluster descriptor + the arch SoC descriptor
    (fabric_all_gather's placement helpers, without its probe dispatch -- the channel is known on the host)."""
    if XPORT_PLACEMENT != "eth" or os.environ.get("TT_METAL_EMULE_MODE"):
        return None
    try:
        from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as fag
    except Exception:  # pragma: no cover - example module unavailable
        return None

    def place(coord, chan, taken, allowed):
        eth_list = fag._chip_maps(mesh_device, coord)[0]
        if not eth_list or chan >= len(eth_list):
            return None
        return fag._worker_nearest_noc1(mesh_device, coord, eth_list[chan], taken, allowed)

    return place


def _plan_placement(mesh_device, num_links, groups=None, links=None):
    """Transport row(s) = the first ceil(4L / grid_x) grid rows (adjacent to the Ethernet row); per chip, each port
    (direction, link) goes to the free transport-row core nearest (NoC1 hops) its Ethernet core, the finals to the
    remaining cores (lowest columns first). A port without a neighbour (line end) still gets a core (it runs the
    receive side). Falls back to the fixed layout [fwd l, bwd l, final (l,0), final (l,1)] per link."""
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    # F finals per link, but never more than fit in the transport row(s) the 2-finals layout needs anyway: an extra
    # transport row would cost a whole compute row (e.g. F=3, L=2 needs 10 cores: fits an 11-wide grid, not 8-wide)
    F = max(1, FINALS_PER_LINK)
    while F > 2 and _cdiv((2 + F) * num_links, gx) > _cdiv(4 * num_links, gx):
        F -= 1
    n_xport = (2 + F) * num_links
    t_rows = _cdiv(n_xport, gx)
    if t_rows >= gy:
        raise ValueError("matmul_reduce_scatter: the core grid has no rows left for compute")
    row_cores = [(x, y) for y in range(t_rows) for x in range(gx)]
    pl = Placement(grid_x=gx, grid_y=gy, transport_rows=t_rows, finals_per_link=F)
    place = _eth_placer(mesh_device) if groups is not None else None
    for coord, (_, prev, nxt) in (groups or {None: (0, None, None)}).items():
        fwd, bwd, taken = [None] * num_links, [None] * num_links, set()
        if place is not None:
            for ports, peer in ((fwd, nxt), (bwd, prev)):
                if peer is None:
                    continue
                for l in range(num_links):
                    core = place(
                        coord, _eth_channel(mesh_device, coord, peer, links[(coord, peer)][l]), taken, row_cores
                    )
                    if core is not None:
                        ports[l] = core
                        taken.add((core.x, core.y))
        if any(c is not None for c in fwd + bwd):
            pl.mode = "eth"
            free = iter(ttnn.CoreCoord(x, y) for x, y in row_cores if (x, y) not in taken)
            fwd = [c if c is not None else next(free) for c in fwd]
            bwd = [c if c is not None else next(free) for c in bwd]
            fin = [next(free) for _ in range(F * num_links)]  # final (l, h) at index l F + h
        else:  # fixed layout (identical on every chip)
            xport = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n_xport)]
            fwd = [xport[(2 + F) * l] for l in range(num_links)]
            bwd = [xport[(2 + F) * l + 1] for l in range(num_links)]
            fin = [xport[(2 + F) * l + 2 + h] for l in range(num_links) for h in range(F)]
        pl.fwd_ports[coord], pl.bwd_ports[coord], pl.finals[coord] = fwd, bwd, fin
    virt = lambda x, y: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    pl.vx = [int(virt(x, 0).x) for x in range(gx)]
    pl.vy = [int(virt(0, y).y) for y in range(gy)]
    # the hand-off gather addresses a compute core by (m-line coord, n-line coord): coordinates must be separable
    for y in range(gy):
        for x in range(gx):
            v = virt(x, y)
            assert (int(v.x), int(v.y)) == (pl.vx[x], pl.vy[y]), "non-separable worker coordinates"
    return pl


def compute_core(pl: Placement, blk: Blocking, ml, nl):
    """Logical core of (m-line, n-line)."""
    if blk.orientation == "A":
        return ttnn.CoreCoord(nl, pl.transport_rows + ml)
    return ttnn.CoreCoord(ml, pl.transport_rows + nl)


def compute_rect(pl: Placement, blk: Blocking):
    nx, ny = (blk.n_lines, blk.m_lines) if blk.orientation == "A" else (blk.m_lines, blk.n_lines)
    return ttnn.CoreRange(ttnn.CoreCoord(0, pl.transport_rows), ttnn.CoreCoord(nx - 1, pl.transport_rows + ny - 1))


# ======================================================================================================================
# Program descriptor
# ======================================================================================================================


def _cset(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _cb(index, num_pages, page_bytes, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _dm(risc, noc):
    return ttnn.DataMovementConfigDescriptor(processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc)


def create_mesh_program_descriptor(
    mesh_device,
    *,
    a,
    w,
    scratch,
    handoff,
    output,
    sems,
    blk: Blocking,
    xp: Transport,
    pl: Placement,
    groups,
    cluster_axis,
    links,
    num_links,
    compute_config,
    ring=False,
):
    """groups: {coord: (p, prev_coord|None, next_coord|None)} (ring: wrap neighbours included);
    links: {(coord, peer): [link ids]}; ring: Topology.Ring schedule (design R3)."""
    sem_arr_fwd, sem_arr_bwd, sem_ready_fence, sem_block_ready = sems[:4]
    sem_block_ack = sems[4:7]  # one ack counter per consumer kind: fwd ports, bwd ports, finals
    G = blk.G
    L = num_links
    F = pl.finals_per_link
    FL = F * L  # final cores per chip
    defs = _PERF_DEFINES + [("MMRS_FINALS_PER_LINK", str(F))]
    cm, cn = blk.core_m_tiles, blk.core_n_tiles
    block_tiles = cm * cn
    rect = compute_rect(pl, blk)
    rect_set = ttnn.CoreRangeSet([rect])
    n_cc = blk.m_lines * blk.n_lines
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    virt = lambda c: (pl.vx[c.x], pl.vy[c.y])
    packed = lambda c: (pl.vx[c.x] << 16) | pl.vy[c.y]

    num_banks = int(mesh_device.dram_grid_size().x * mesh_device.dram_grid_size().y)  # A / W interleave banks
    a_ct = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    scr_ct = list(ttnn.TensorAccessorArgs(scratch).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    a_addr, w_addr = int(a.buffer_address()), int(w.buffer_address())
    scr_addr, out_addr = int(scratch.buffer_address()), int(output.buffer_address())
    handoff_base = int(handoff.buffer_address())

    W = blk.waves
    a_pages = cm * blk.Kt if blk.a_resident else OPERAND_DEPTH * cm * blk.k_block_tiles
    w_pages = blk.Kt * cn if blk.w_resident else OPERAND_DEPTH * blk.k_block_tiles * cn
    um, un = blk.unit_m_tiles, blk.unit_n_tiles
    rows_wave = blk.scatter_dim == -2  # waves split the scatter axis: rows (-2) or columns (-1)
    win = [
        (
            Window(wv * um, 0, (wv + 1) * um, 0, xp.segs_per_row)
            if rows_wave
            # (a column-wave boundary is a whole segment; the block's last segment of a row may be ragged: ceil)
            else Window(0, wv * un, blk.blk_m_tiles, wv * un // xp.seg_tiles, _cdiv((wv + 1) * un, xp.seg_tiles))
        )
        for wv in range(W)
    ]
    xport_pages = xp.cap_segs * xp.seg_tiles
    # transport add block: the DEST batch, within one segment (the add walks segments x seg_tiles, row-blocked, so a
    # segment is never held back for the next one's tiles and no block straddles the CB wrap)
    xport_add_block = min(XPORT_ADD_BLOCK_MAX, xp.seg_tiles)
    sched = {q: _schedule_mmrs(q, G, ring) for q in range(G)}

    # compute-rectangle virtual bounds for the ack multicast (NoC0: start = min corner)
    x0, y0 = virt(rect.start)
    x1, y1 = virt(rect.end)
    m_on_y = 1 if blk.orientation == "A" else 0
    mcoords = [pl.vy[pl.transport_rows + ml] if m_on_y else pl.vx[ml] for ml in range(blk.m_lines)]
    ncoords = [pl.vx[nl] if m_on_y else pl.vy[pl.transport_rows + nl] for nl in range(blk.n_lines)]
    # NoC of each operand's injector reads + line multicast (A: NCRISC, W: BRISC); MMRS_A_NOC / MMRS_W_NOC override
    # "auto": the block-invariant operand (A for scatter_dim=-1, W for -2; loaded during block 0, while the transport is
    # idle) reads on NoC0, the operand streamed through the whole op reads on NoC1, off the transport's NoC0 traffic
    a_invariant = blk.scatter_dim == -1
    a_noc = {"0": NOC0, "1": NOC1}.get(os.environ.get("MMRS_A_NOC", "auto"), NOC0 if a_invariant else NOC1)
    w_noc = {"0": NOC0, "1": NOC1}.get(os.environ.get("MMRS_W_NOC", "auto"), NOC1 if a_invariant else NOC0)
    # Line-injector placement (MMRS_INJ): "first" = the first core of every line (A injectors form one grid column or
    # row, W injectors the other); "diag" = the sender advances one core per line (A: m-line ml injects from n-line
    # (ml + A_INJ_START) mod n_lines; W: n-line nl from m-line (nl + W_INJ_START) mod m_lines), so no two injectors of
    # an operand share a grid row or column and their DRAM reads never converge on one NoC row / column.
    diag = os.environ.get("MMRS_INJ", "diag") == "diag"
    a_inj_start, w_inj_start = 0, (1 if diag else 0)
    a_sender_nl = lambda ml: (a_inj_start + ml) % blk.n_lines if diag else 0
    w_sender_ml = lambda nl: (w_inj_start + nl) % blk.m_lines if diag else 0
    sender_placement = ttnn.Mcast1DSenderPlacement.Diagonal if diag else ttnn.Mcast1DSenderPlacement.Uniform
    a_line_shape = ttnn.Mcast1DShape.PerRow if blk.orientation == "A" else ttnn.Mcast1DShape.PerColumn
    w_line_shape = ttnn.Mcast1DShape.PerColumn if blk.orientation == "A" else ttnn.Mcast1DShape.PerRow
    # per-chip placement (pl.mode "eth") or one shared layout (keyed None)
    at = lambda ports, coord: ports[coord] if coord in ports else ports[None]

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, (p, prev, nxt) in groups.items():
        fwd, bwd, order, fwd_up, bwd_up = sched[p]
        my_fwd, my_bwd, my_finals = at(pl.fwd_ports, coord), at(pl.bwd_ports, coord), at(pl.finals, coord)
        consumers = [packed(c) for c in my_fwd + my_bwd + my_finals]
        # compute units: each block of the order expanded into its waves 0..W-1
        units = [(j, wv) for j in order for wv in range(W)]
        cidx = {u: i for i, u in enumerate(units)}
        program = ttnn.ProgramDescriptor()
        kernels = []

        # ---------------- compute rectangle ----------------
        cbs = [
            _cb(CB_ACT_OPERAND, a_pages, blk.a_tile_bytes, a.dtype, rect_set),
            _cb(CB_WEIGHT_OPERAND, w_pages, blk.w_tile_bytes, w.dtype, rect_set),
            _cb(CB_PARTIAL_ACCUM, block_tiles, blk.acc_tile_bytes, blk.acc_dtype, rect_set),
            ttnn.cb_descriptor_from_sharded_tensor(CB_PARTIAL_HANDOFF, handoff),
        ]
        # per compute unit: A row origin, consumer kind (0 fwd ports, 1 bwd ports, 2 finals), cumulative acks, A fresh
        # (read, vs. replayed from the resident ring); W: column origin, W fresh
        order_rt, w_order_rt, cum = [], [], [0, 0, 0]  # cumulative acks per consumer kind (each kind acks in order)
        for b, (j, wv) in enumerate(units):
            kind = 2 if j == p else (0 if j in fwd else 1)
            cum[kind] += FL if kind == 2 else L
            a_row = (j * blk.blk_m_tiles if rows_wave else 0) + win[wv].r0
            order_rt += [a_row, kind, cum[kind], int(not (blk.a_resident and b > 0))]
            w_col = (0 if rows_wave else j * blk.blk_n_tiles) + win[wv].c0
            w_order_rt += [w_col, int(not (blk.w_resident and b > 0))]

        a_groups = {1: [], 0: []}
        w_groups = {2: [], 1: [], 0: []}
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for ml in range(blk.m_lines):
            for nl in range(blk.n_lines):
                c = compute_core(pl, blk, ml, nl)
                a_groups[int(nl == a_sender_nl(ml))].append(c)
                w_groups[2 if W_ROT else int(ml == w_sender_ml(nl))].append(c)
                row0 = ml * cm
                col0 = nl * cn
                rd_rt[c.x][c.y] = (
                    [
                        a_addr,
                        blk.Kt,
                        row0,
                        min(cm, um - row0),
                        sem_block_ready,
                        *sem_block_ack,
                    ]
                    + order_rt
                    + consumers
                )
                wr_rt[c.x][c.y] = (
                    [
                        w_addr,
                        blk.Nt,
                        col0,
                        min(cn, un - col0),
                    ]
                    + w_order_rt
                    + ([nl] if W_ROT else [])
                )

        reader_kernels, writer_kernels = [], []
        for sends, cores in a_groups.items():
            if not cores:
                continue
            rt = ttnn.RuntimeArgs()
            for c in cores:
                rt[c.x][c.y] = rd_rt[c.x][c.y]
            reader_kernels.append(
                ttnn.KernelDescriptor(
                    defines=defs,
                    kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_reader.cpp"),
                    core_ranges=_cset(cores),
                    compile_time_args=[
                        CB_ACT_OPERAND,
                        CB_PARTIAL_HANDOFF,
                        cm,
                        blk.k_block_tiles,
                        blk.num_k_blocks,
                        len(units),
                        block_tiles,
                        blk.a_tile_bytes,
                        sends,
                        L,
                        INJECT_READ_AHEAD,
                        num_banks,
                    ]
                    + a_ct,
                    runtime_args=rt,
                    config=_dm("RISCV_1", a_noc),
                )
            )
        for sends, cores in w_groups.items():
            if not cores:
                continue
            rt = ttnn.RuntimeArgs()
            for c in cores:
                rt[c.x][c.y] = wr_rt[c.x][c.y]
            writer_kernels.append(
                ttnn.KernelDescriptor(
                    defines=defs,
                    kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_writer.cpp"),
                    core_ranges=_cset(cores),
                    compile_time_args=[
                        CB_WEIGHT_OPERAND,
                        cn,
                        blk.k_block_tiles,
                        blk.num_k_blocks,
                        len(units),
                        blk.w_tile_bytes,
                        sends,
                        INJECT_READ_AHEAD,
                        int(blk.Nt % num_banks == 0),
                    ]
                    + w_ct,
                    runtime_args=rt,
                    config=_dm("RISCV_0", w_noc),
                )
            )
        mc_a = ttnn.Mcast1D(
            mesh_device,
            rect_set,
            a_line_shape,
            ttnn.Mcast1DFixedSenderConfig(starting_sender_index=a_inj_start, sender_placement=sender_placement),
            ttnn.McastConfig(noc=a_noc, handshake=True),
        )
        mc_w = ttnn.Mcast1D(
            mesh_device,
            rect_set,
            w_line_shape,
            ttnn.Mcast1DRotatingSenderConfig()
            if W_ROT
            else ttnn.Mcast1DFixedSenderConfig(starting_sender_index=w_inj_start, sender_placement=sender_placement),
            ttnn.McastConfig(noc=w_noc, handshake=True),
        )
        mc_a.attach(program, "a", reader_kernels)
        mc_w.attach(program, "w", writer_kernels)
        kernels += reader_kernels + writer_kernels
        kernels.append(
            ttnn.KernelDescriptor(
                defines=defs,
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_compute.cpp"),
                core_ranges=rect_set,
                compile_time_args=[
                    CB_ACT_OPERAND,
                    CB_WEIGHT_OPERAND,
                    CB_PARTIAL_ACCUM,
                    CB_PARTIAL_HANDOFF,
                    cm // blk.out_subblock_h,
                    cn // blk.out_subblock_w,
                    blk.out_subblock_h,
                    blk.out_subblock_w,
                    blk.k_block_tiles,
                    blk.num_k_blocks,
                    len(units),
                ],
                runtime_args=[],
                config=compute_config,
            )
        )

        # ---------------- transport row ----------------
        line_rt = [m for m in mcoords] + [n for n in ncoords]

        def xport_entries(blocks, ups, slot_a, slot_b, first, stride):
            """Reader entries, one per (block, wave) of the list: [hand-off slot, scratch slot A, scratch slot B,
            has_upstream, first seg, seg count, wave tile origin r0, c0, segment columns s0, s1] -- the core's
            segments of the wave's window."""
            return [
                (
                    cidx[(j, wv)] % blk.handoff_slots,
                    slot_a(j),
                    slot_b(j),
                    int(u),
                    *win[wv].segs(first, stride, xp.segs_per_row),
                    win[wv].r0,
                    win[wv].c0,
                    win[wv].s0,
                    win[wv].s1,
                )
                for j, u in zip(blocks, ups)
                for wv in range(W)
            ]

        def xport_reader_rt(stride, entries, arr_a, arr_b, ack_sem):
            return (
                [
                    scr_addr,
                    handoff_base,
                    block_tiles * BF16_TILE_BYTES,
                    xp.segs_per_block,
                    xp.segs_per_row,
                    blk.blk_n_tiles,
                    stride,
                    sem_block_ready,
                    n_cc,
                    arr_a,
                    arr_b,
                    ack_sem,
                    *((x1, y1, x0, y0) if XPORT_NOC_SWAP else (x0, y0, x1, y1)),  # NoC1 multicast: start = max corner
                    blk.m_lines,
                    blk.n_lines,
                    len(entries),
                ]
                + [v for e in entries for v in e]
                + line_rt
            )

        def xport_reader_kernel(cores, rt, target, has_a, has_b):
            return ttnn.KernelDescriptor(
                defines=defs,
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_xport_reader.cpp"),
                core_ranges=_cset(cores),
                compile_time_args=[
                    target,
                    CB_XPORT_ARRIVAL_A,
                    CB_XPORT_ARRIVAL_B,
                    has_a,
                    has_b,
                    xp.seg_tiles,
                    xp.xport_group,
                    INC_EVERY,
                    cm,
                    cn,
                    m_on_y,
                    xp.cap_segs,
                    BF16_TILE_BYTES,
                ]
                + scr_ct,
                runtime_args=rt,
                config=_dm("RISCV_1", NOC1 if XPORT_NOC_SWAP else NOC0),
            )

        def add_kernel(cores, rt, has_a, has_b):
            return ttnn.KernelDescriptor(
                defines=defs,
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_xport_add.cpp"),
                core_ranges=_cset(cores),
                compile_time_args=[
                    CB_XPORT_PARTIAL,
                    CB_XPORT_ARRIVAL_A,
                    CB_XPORT_ARRIVAL_B,
                    CB_XPORT_SUM,
                    has_a,
                    has_b,
                    xport_add_block,
                    xp.seg_tiles,
                    CB_XPORT_ZERO,
                ],
                runtime_args=rt,
                # every cross-device addition accumulates in fp32 (requirement, independent of the user's config)
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            )

        relay_ports, end_ports, finals = [], [], list(my_finals)
        rd_relay, rd_end, add_relay, snd_rt = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        senders = []
        for d, blocks, ups, peer, port_list in (
            ("fwd", fwd, fwd_up, nxt, my_fwd),
            ("bwd", bwd, bwd_up, prev, my_bwd),
        ):
            if peer is None:
                continue  # no neighbour in this direction: nothing to send, nobody sends a ready to it
            assert blocks, "a port with a neighbour always has blocks to send (G >= 2 line, G >= 3 ring)"
            assert ups == sorted(ups), "upstream-less entries must precede relay entries"
            n_up = sum(ups)
            relay = n_up > 0  # mixed (ring) or all-relay (line interior): the add kernel produces cb_xport_sum
            peer_p = groups[peer][0]
            # the peer chip's cores (its own placement): opposite-direction port (ready fence), same-direction port
            # (relay arrivals), finals (own-block arrivals)
            peer_opp = at(pl.bwd_ports if d == "fwd" else pl.fwd_ports, peer)
            peer_same = at(pl.fwd_ports if d == "fwd" else pl.bwd_ports, peer)
            peer_finals = at(pl.finals, peer)
            for l in range(L):
                core = port_list[l]
                entries = xport_entries(blocks, ups, lambda j: j, lambda j: 0, l, L)
                rt = xport_reader_rt(L, entries, sem_arr_fwd, sem_arr_fwd, sem_block_ack[0 if d == "fwd" else 1])
                (rd_relay if relay else rd_end)[core.x][core.y] = rt
                (relay_ports if relay else end_ports).append(core)
                if relay:  # [segments copied through (upstream-less entries), segments added (relay entries)]
                    add_relay[core.x][core.y] = [
                        sum(e[5] for e in entries if not e[3]),
                        sum(e[5] for e in entries if e[3]),
                    ]
                # sender entries per (block, wave): [landing slot (j, except the receiver's own block arriving
                # backward -> slot G), first seg, count, for the downstream finals (the last block: the downstream
                # chip's own), their per-half counts in the wave (0 / 0 for relay blocks)]
                snd_entries = []
                for j, up in zip(blocks, ups):
                    fin = j == blocks[-1]
                    for wv in range(W):
                        wn = win[wv]
                        snd_entries += [
                            G if (d == "bwd" and j == peer_p) else j,
                            *wn.segs(l, L, xp.segs_per_row),
                            int(fin),
                            *[wn.segs(l + h * L, FL, xp.segs_per_row)[1] if fin else 0 for h in range(F)],
                            wn.s0,
                            wn.s1,
                            j,  # this chip's arrival slot of block j (the relay reader's base_a)
                            int(bool(up)),
                        ]
                expect_in = sum(_incs(e[5]) for e in entries if e[3])  # arrival increments upstream sends here
                rc = virt(peer_opp[l])  # peer chip's opposite-direction port
                pc = virt(peer_same[l])  # downstream port of the same (direction, link)
                fins = [virt(peer_finals[F * l + h]) for h in range(F)]
                pn = node(peer)
                args = [
                    scr_addr,
                    xp.segs_per_block,
                    xp.segs_per_row,
                    blk.blk_n_tiles,
                    L,
                    sem_arr_fwd,
                    sem_arr_fwd if d == "fwd" else sem_arr_bwd,
                    expect_in,
                    sem_ready_fence,
                    1,
                    rc[0],
                    rc[1],
                    pc[0],
                    pc[1],
                    *[v for f in fins for v in (f[0], f[1])],
                    int(pn.mesh_id),
                    int(pn.chip_id),
                    len(blocks) * W,
                ] + snd_entries
                args += list(ttnn.setup_fabric_connection(node(coord), pn, links[(coord, peer)][l], program, core))
                snd_rt[core.x][core.y] = args
                senders.append(core)

        # finals receive the own block forward iff the previous chip's fwd list (ending with p) is non-empty, backward
        # iff the next chip's bwd list is
        has_fa = int(prev is not None and len(sched[groups[prev][0]][0]) > 0)
        has_fb = int(nxt is not None and len(sched[groups[nxt][0]][1]) > 0)
        rd_final, add_final, wr_final = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(finals):  # final (l, h) = index l F + h: segments l + h L, stride F L
            first = i // F + (i % F) * L
            entries = xport_entries([p], [True], lambda j: j, lambda j: G, first, FL)
            rd_final[core.x][core.y] = xport_reader_rt(FL, entries, sem_arr_fwd, sem_arr_bwd, sem_block_ack[2])
            add_final[core.x][core.y] = [0, sum(e[5] for e in entries)]
            incs = sum(_incs(e[5]) for e in entries)  # one increment stream per wave (see the port sender)
            # the summed own block leaves the add kernel wave by wave: the writer walks the same windows
            wr_entries = [v for e in entries for v in (e[4], e[5], e[8], e[9])]
            wr_final[core.x][core.y] = [
                out_addr,
                xp.segs_per_row,
                blk.blk_n_tiles,
                FL,
                sem_arr_fwd,
                incs if has_fa else 0,
                sem_arr_bwd,
                incs if has_fb else 0,
                len(entries),
            ] + wr_entries

        xport_cores = relay_ports + end_ports + finals
        cbs.append(_cb(CB_XPORT_SUM, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(xport_cores)))
        if relay_ports + finals:
            cbs.append(_cb(CB_XPORT_PARTIAL, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(relay_ports + finals)))
        arr_a_cores = relay_ports + (finals if has_fa else [])
        if arr_a_cores:
            cbs.append(_cb(CB_XPORT_ARRIVAL_A, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(arr_a_cores)))
        if has_fb:
            cbs.append(_cb(CB_XPORT_ARRIVAL_B, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(finals)))
        if has_fa and has_fb:
            cbs.append(_cb(CB_XPORT_ZERO, 1, BF16_TILE_BYTES, ttnn.bfloat16, _cset(finals)))

        if relay_ports:
            kernels.append(xport_reader_kernel(relay_ports, rd_relay, CB_XPORT_PARTIAL, 0 if PORT_ARR else 1, 0))
            kernels.append(add_kernel(relay_ports, add_relay, 1, 0))
        if end_ports:
            kernels.append(xport_reader_kernel(end_ports, rd_end, CB_XPORT_SUM, 0, 0))
        kernels.append(xport_reader_kernel(finals, rd_final, CB_XPORT_PARTIAL, has_fa, has_fb))
        kernels.append(add_kernel(finals, add_final, has_fa, has_fb))
        if senders:
            kernels.append(
                ttnn.KernelDescriptor(
                    defines=defs + ([("MMRS_PORT_ARR_CB", str(CB_XPORT_ARRIVAL_A))] if PORT_ARR else []),
                    kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_port_sender.cpp"),
                    core_ranges=_cset(senders),
                    compile_time_args=[
                        CB_XPORT_SUM,
                        xp.seg_tiles,
                        xp.xport_group,
                        INC_EVERY,
                        xp.cap_segs,
                        BF16_TILE_BYTES,
                    ]
                    + scr_ct,
                    runtime_args=snd_rt,
                    config=_dm("RISCV_0", NOC0 if XPORT_NOC_SWAP else NOC1),
                )
            )
        kernels.append(
            ttnn.KernelDescriptor(
                defines=defs,
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_final_writer.cpp"),
                core_ranges=_cset(finals),
                compile_time_args=[CB_XPORT_SUM, xp.seg_tiles, xp.xport_group, xp.cap_segs, BF16_TILE_BYTES] + out_ct,
                runtime_args=wr_final,
                config=_dm("RISCV_0", NOC0 if XPORT_NOC_SWAP else NOC1),
            )
        )
        program.cbs = cbs
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
