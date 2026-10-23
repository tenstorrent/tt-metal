# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm MeshProgramDescriptor (op_design.md -> Blocking Model / Work Distribution).

Per chip (identical kernels on every chip; only runtime args differ):
  port cores  2·L_act (one per (direction, link)), logical row 0, centred — or, when a sharded input's shard grid
              covers that row, the last free cores of the grid (reverse row-major; a port never shares a core with the
              reader / compute / writer trio, whose RISCV_0 is taken). One data-movement kernel (RISCV_0, NoC1):
              ready fence (one ready from every destination chip), then per wave: own segment out — straight to every
              peer on a Ring (DIRECT_RING_SENDS, multi-hop), or to the neighbour plus relays (Linear) — and a notify
              of the roots per landed entry (NOTIFY_PER_ENTRY; one arrival counter per entry). _exchange_plan.
              Only moves bytes. Their only L1 is the persistent landing scratch (mesh tensor, same address on
              every chip) and the fabric header pool.
  compute     every other Tensix core. Local regimes from the base op rms_norm:
              R1-X  rows split over the grid (split_work_to_cores semantics), whole rows resident, every core a root;
              R3-X  row-groups of R3_ROWS_PER_GROUP tile-rows × G_w W-slice cores, root = the group's first core,
                    in-chip gather of the core partials into the root; no inv distribution: the root forwards the
                    raw stat slots (own at publish, each peer slot as it lands) to its members by linked multicast
                    over the member rectangles (row-groups are single rectangles where possible: RECT_GROUPS,
                    _rect_groups), and every member finalizes its own inv (FWD_INV, Perf 1). R3-RS (several owners)
                    keeps the owners' unicast inv all-gather.
              Sharded inputs (native, zero-copy: cb_x_tiles / cb_output backed on the input / output shard; no x or
              output DRAM / NoC traffic, the shard spec IS the work split, the block is the whole shard clamped by L1):
              HEIGHT_SHARDED -> R1-X, one group per shard core with that shard's tile-rows;
              WIDTH_SHARDED  -> R3-X, ONE row-group = the shard cores in shard order, G_w = #slices, S = shard width,
                                its rows split over R3-RS owners (R3_RS_OWNERS), each an independent root.
              Roots publish their device partial (compact fp32, 128 B per tile-row) to their link's ports, gather the
              G-1 peer partials from the ports' landing, sum all G in device order (fp32 SFPU) and finalize.

Knobs (single source of truth, all module constants; every dependent quantity derives from them):
  BLOCK_ROWS_CAP       cap on block_rows R (None = uncapped). Read at call time (the acceptance test monkeypatches it).
                       R = min(max_group_rows, cap, R_fit); num_waves = ceil(max_group_rows / R).
  MAX_PIPELINE_DEPTH   x_depth = min(MAX_PIPELINE_DEPTH, num_waves) (gather / inv depth: 1, see _depths)
  WAVE_PIPELINE        num_waves > 1 (and the footprint fits, see _depths): one-block look-ahead across waves (reader: load / combine / publish of wave
                       w+1 before wave w's slot gather + inv distribution; compute: square / collapse / combine of
                       w+1 before wave w's finalize + scale), so wave w's exchange overlaps wave w+1's DRAM reads and
                       front half; gather / stats-slot depths 1 -> 2, interleaved x ring = every wave (see _depths).
                       False = waves run back to back.
  R3_MIN_WAVES         interleaved R3-X row-groups only: the block cap the plan applies by itself,
                       R <= ceil(max_group_rows / R3_MIN_WAVES) (1 = one block when it fits). With WAVE_PIPELINE, > 1
                       trades a second exchange round for the overlap (see the measured basis at the constant).
  STREAM_DEPTH         cb_output depth in C-tile quanta
  W_CHUNK_TILES        streaming chunk cap; C = ceil(S / ceil(S / W_CHUNK_TILES))
  R3_*                 interleaved R3-X layout and selection (base op values)
  MAX_HEADERS_IN_FLIGHT packet headers per port: one pre-routed header per destination chip (<= 12 per RISC)
  DIRECT_RING_SENDS / NOTIFY_PER_ENTRY   exchange transport levers L6 / L4 (Refinement 5)
  RECT_GROUPS          interleaved R3-X row-groups as hole-free rectangles (fewest slot-forward multicasts)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32
FACE_DIM = 16

# ---- Knobs ----
BLOCK_ROWS_CAP = None
MAX_PIPELINE_DEPTH = 2
WAVE_PIPELINE = True
STREAM_DEPTH = 2
W_CHUNK_TILES = 64
# L1 fallbacks (l1_fallbacks()), tried in order when the default layout (resident γ window, W_CHUNK_TILES chunks) leaves no R >= 1 —
# a wide shard (in + out shards of 0.5-1 MB each) leaves ~0.4 MB of CB space: (chunk cap, γ resident). A streamed γ
# holds GAMMA_STREAM_DEPTH chunks of C pages (re-read per block: row-0 prefixes only) instead of the WS-page window.
GAMMA_STREAM_DEPTH = 2


def l1_fallbacks():
    """(chunk cap, γ resident) layouts in the order the solve tries them (derived from W_CHUNK_TILES at call time)."""
    return [(W_CHUNK_TILES, True)] + [(W_CHUNK_TILES >> k, False) for k in range(W_CHUNK_TILES.bit_length())]


R3_ROWS_PER_GROUP = 2
R3_MAX_GROUP_CORES = 24
R3_MIN_SLICE_TILES = 4
R3_OCCUPANCY_GAIN = 2
R3_MIN_GROUP_CORES = 4
# Interleaved R3-X: split each row-group's rows into >= R3_MIN_WAVES blocks (the wave look-ahead then overlaps block
# w's exchange + inv distribution with block w+1's DRAM reads / combine, and block w's output writes with block w+1's
# exchange). Measured, traced 1x4 ring, 2 links, bf16, slowest device: 640x1792 (10 row-groups of 2 rows x 10 cores)
# 27.3 -> 22.3 us. Not applied elsewhere: interleaved R1-X (4096x1024 57.2 -> 59.2 us, 8192x1024 99.2 -> 102.0 us:
# every core is its own root, so no in-chip chain to hide) and WIDTH_SHARDED R3-RS (640x1792 25.8 -> 31.0 us).
R3_MIN_WAVES = 2
# R3-RS (in-chip reduce-scatter owners; WIDTH_SHARDED row-groups, which hold ALL rows): the block's R tile-rows are
# split over `owners` cores of the row-group, owner k sums / publishes / finalizes rows [k*q, k*q + q) and unicasts
# their inv to every member. owners = the largest divisor of R <= min(G_w, R3_RS_OWNERS) (None = G_w; 1 = a single
# root, the interleaved R3-X shape), so every owner has exactly q = R / owners rows in every full block (uniform CB
# quanta: no push ever straddles a ring end). Interleaved R3-X row-groups hold R3_ROWS_PER_GROUP rows: single root.
R3_RS_OWNERS = None
MAX_HEADERS_IN_FLIGHT = 8
# Port placement (lamp L3): first logical x of the 2·L_act port cores on row 0 (None = centred). Measured flat on the
# 640x1792 ring (packets of ~1 KB are latency-, not bandwidth-bound): kept as a knob.
PORT_X0 = None
# Exchange transport on a Ring of G >= 3 (lamp L6): True = every chip sends its own segment straight to every peer
# (multi-hop unicast routed by destination node, first hop on the port whose direction
# ttnn.get_eth_forwarding_direction reports), no relay-kernel turnaround; False = the one-hop store-and-forward relay
# (Linear always uses the relay: line neighbours only).
DIRECT_RING_SENDS = True
# Notify granularity (lamp L4): True = a port notifies its roots per landed entry and a root reads each slot as soon as it
# lands; False = once per wave, after the wave's last entry.
NOTIFY_PER_ENTRY = True
# Row-group shape (interleaved R3-X): True = hole-free rectangles (_rect_groups: row segments, then column segments of
# what is left; port cores are never members), so a root's slot forward (FWD_INV) is one multicast chain per group;
# False = consecutive compute cores (row runs: several rectangles per group). Both are correct (the forward takes any
# number of rectangles); Refinement 5 measured the single rectangle faster for the multicast it then used.
RECT_GROUPS = True
# NoC balance (lamp L8): (reader NoC, writer NoC) of the compute cores; (0, 1) = the Reader/Writer config defaults.
# Measured on the 640x1792 ring (2 waves): (0, 1) 22.6 us, (1, 0) 32.2 us (the output DRAM writes on NoC0 cross the
# grid against the reads). Both on one NoC is not a valid dedicated-NoC configuration (the two RISCs would share the
# NoC's transaction counters): refused below.
DM_NOCS = (0, 1)
# DRAM NoC balance (Perf 1, noc_balanced_writes; the base op's Perf 2 dram_noc_balance): the compute cores of the
# NOC_SWAP_COLUMNS westmost logical columns (on Blackhole the Tensix columns right next to DRAM column 0) and of the
# east-most logical column run their x reads on the writer's NoC and their output writes on the reader's NoC (a second
# reader / writer kernel descriptor pair, same sources and args, DM_DEDICATED_NOC: each RISC still owns one NoC).
# With every core on DM_NOCS, the output-write burst of a wave queues on NoC1's routes from the east column (measured
# on the 640x1792 ring: wave-0 writes 0.3 us on the west cores vs 6-7 us on the east column, the op ending ~4.4 us
# after the last inv). Bench (single chip, the op's geometry): 640x1792 15.1 -> 11.75 us (slowest-core write tail
# 9.0 -> 6.25 us), 2048x1280 -20 %, 4096x1024 -21 %, 1024x4096 -19 %; per-tile / per-bank NoC splits and the read
# twin alone were flat or worse. Carve-outs: the east column swaps only when it holds >= 2 compute cores (a lone core
# there, decode 32x1792, measured +17 % swapped vs -3.6 % without it); multicast senders (FWD_INV roots) never swap
# (their multicast rectangles are packed for the reader's DM_NOCS NoC); a placement with no x / output DRAM traffic
# (input AND output shard-resident) has nothing to balance.
NOC_SWAP_COLUMNS = 3
NOC_SWAP_EAST_MIN_CORES = 2

# ---- Mechanism constants ----
STATS_ROW_BYTES = 32 * 4  # one tile-row's 32 fp32 partials on the wire
SEG_ALIGN = 64  # DRAM / L1 NoC alignment of a landing slot (Blackhole DRAM 64 B covers L1 16 B)
LANDING_MIN_BYTES = 4096  # smallest landing-scratch bucket (per port core); buckets are powers of two above it
SCALER_FORMAT = ttnn.bfloat16
# Stats pages (cb_partial_col, cb_gather, cb_device_partial, cb_stats_slots, cb_inv_root, cb_inv_rms): Float32 whatever
# the DEST width — the reader moves their column 0 / compact 128 B tails as 4-byte words (col0_word), and the exchanged
# partials are fp32 on the wire and summed from an UnpackToDestFp32 CB. One tile per tile-row: negligible L1 / bytes.
STATS_FORMAT = ttnn.float32


def dest_format(fp32_dest_acc_en):
    """Page format of the DEST-following intermediates (cb_sumsq_partial, cb_x_gamma) — the ONE place it is derived:
    Float32 with 32-bit DEST, Float16_b with 16-bit DEST (the value is already rounded to 16 bits in DEST before the
    pack, so a wider page would carry no precision and cost packer / unpacker bytes and L1)."""
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


# ---- CB indices (mirrored by the kernels) ----
CB_X_TILES = 0
CB_REDUCE_SCALER = 1
CB_SUMSQ_PARTIAL = 2
CB_PARTIAL_COL = 3
CB_GATHER = 4
CB_DEVICE_PARTIAL = 5
CB_STATS_SLOTS = 6
CB_INV_ROOT = 7
CB_INV_RMS = 8
CB_GAMMA_TILES = 9
CB_X_GAMMA = 10
CB_OUTPUT = 16

# ---- Program-local semaphores (declared over compute ∪ port cores BEFORE any setup_fabric_connection) ----
SEM_GATHER = 0  # R3 root: partial arrivals from its members
SEM_INV = 1  # R3 member: inv arrivals from its root
SEM_STATS_LANDED = 2  # root: "every slot of wave w has landed" notifications from its link's ports
SEM_OWN_STATS = 3  # port: own-segment rows published by its link's roots
# root: "wave w's publishes are counted" signals from each of its link's sending ports. SEM_OWN_STATS is cumulative over
# waves, so a root whose next block is empty would otherwise increment for wave w+1 while a slower root's wave-w rows
# are still in flight, and the port would ship wave w with stale rows. A root's wave-(w+1) increment waits for
# counted(w) from every sending port (measured: ~40 of 108 rows stale per call on R1 with num_waves = 2 before this).
SEM_OWN_COUNTED = 4
# WAVE_PIPELINE: a member may send block b+1's partials before a slower member sent block b's (the look-ahead), so a
# cumulative count would let b+1 stand in for b. Block b counts on SEM_GATHER_BY_PARITY[b % 2]; a member can never be
# two blocks ahead (its block b+2 partials wait for inv(b), which needed every member's block b).
SEM_GATHER_ODD = 5
SEM_GATHER_BY_PARITY = (SEM_GATHER, SEM_GATHER_ODD)
# Each port notifies its roots once per wave, in wave order, on its OWN semaphore (one per direction): with the
# look-ahead one port may notify wave w+1 before the other port notifies wave w, which a shared count would miss.
SEM_STATS_LANDED_BWD = 6
# FWD_INV (single-root R3-X, Perf 1 replicated_finalize): the root's multicast SETs on every member. SEM_OWN_FWD = blocks
# whose own slot it forwarded; SEM_SLOT_FWD = (block + 1) << 24 | mask of the peer slots forwarded for that block.
SEM_OWN_FWD = 7
SEM_SLOT_FWD = 8

COMMON_CT_ARGS = 28  # length of the CT-arg prefix shared by reader / writer / compute (accessor args follow)

FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}
SEM_LANDED = {FWD: SEM_STATS_LANDED, BWD: SEM_STATS_LANDED_BWD}
_NOCS = (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)


def _fp32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def _pack(c) -> int:
    return (int(c.x) << 16) | int(c.y)


# ----------------------------------------------------------------------------------------------------------------------
# Groups, links, topology (host-only; also used by validate())
# ----------------------------------------------------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def _group_hops(groups, ring):
    hops = []
    for grp in groups:
        G = len(grp)
        for i in range(G if ring else G - 1):
            a, b = grp[i], grp[(i + 1) % G]
            hops += [(a, b), (b, a)]
    return hops


def links_available(mesh, cluster_axis, ring):
    """L_avail: min over every hop the topology uses (both directions, the wrap hop for Ring)."""
    groups = build_groups(tuple(mesh.shape), cluster_axis)
    return min(
        len(ttnn.get_forwarding_link_indices(_node(mesh, a), _node(mesh, b))) for a, b in _group_hops(groups, ring)
    )


def check_ring_wrap(mesh, cluster_axis):
    """Ring with G >= 3: the active fabric config must wrap cluster_axis and every wrap hop must have links."""
    cfg = ttnn.get_fabric_config()
    F = ttnn.FabricConfig
    wraps = {
        F.FABRIC_2D_TORUS_XY: (0, 1),
        F.FABRIC_2D_TORUS_X: (1,),
        F.FABRIC_2D_TORUS_Y: (0,),
        F.FABRIC_1D_RING: (0, 1),
    }.get(cfg, ())
    if cluster_axis not in wraps:
        raise ValueError(
            f"distributed_rms_norm: Ring along cluster_axis={cluster_axis} needs a fabric config that wraps it (got {cfg})"
        )
    for grp in build_groups(tuple(mesh.shape), cluster_axis):
        a, b = grp[-1], grp[0]
        for s, d in ((a, b), (b, a)):
            if not ttnn.get_forwarding_link_indices(_node(mesh, s), _node(mesh, d)):
                raise ValueError(f"distributed_rms_norm: Ring wrap hop {s} -> {d} has no link")


def _schedule(p, G, ring):
    """(slots sent toward p+1, slots sent toward p-1), own slot first, in arrival order downstream (AG _schedule)."""
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [(p - i) % G for i in range(kf)], [(p + i) % G for i in range(kb)]
    fwd = [p - i for i in range(p + 1)] if p < G - 1 else []
    bwd = [p + i for i in range(G - p)] if p > 0 else []
    return fwd, bwd


def _first_hop_dir(mesh, src, dst, nxt, prv):
    """FWD / BWD: the port of `src` whose connection carries the first hop of the fabric's route src -> dst (None when
    the route leaves through neither ring neighbour)."""
    d = ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, dst))
    if d is None:
        return None
    for port, nb in ((FWD, nxt), (BWD, prv)):
        if d == ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, nb)):
            return port
    return None


def _exchange_plan(mesh, groups_coords, ring):
    """Per chip coord: {p, next, prev, direct, sends, entries, ready_out} — the stats all-gather's transport.
      sends[d]     items of port d, in send order: (slot q, destination coord, destination entry index, hops,
                   wait) — wait 0 = the own segment, j + 1 = relay of this port's entry j once it landed
      entries[d]   slots landing on port d, in entry order (entry i has arrival counter i)
      ready_out[d] ready packets port d sends: (target coord, target port direction, hops)
    Direct (Ring, G >= 3, DIRECT_RING_SENDS, every route leaves through a ring neighbour): every item is the own
    segment to one peer, the port picked by the route's first hop, nearest peer first; a port's entries are its
    sources ordered by distance (the expected arrival order). Relay: _schedule (AG): own segment to the neighbour, then
    every upstream entry but the last relayed on. The ready fence: every chip a port writes into sends that port one
    ready (on its own port toward the sender, by the route's first hop)."""
    chips = {}
    for grp in groups_coords:
        G = len(grp)
        nxt = [grp[(p + 1) % G] if (ring or p < G - 1) else None for p in range(G)]
        prv = [grp[(p - 1) % G] if (ring or p > 0) else None for p in range(G)]
        info = [
            dict(
                p=p,
                next=nxt[p],
                prev=prv[p],
                direct=False,
                sends={FWD: [], BWD: []},
                entries={FWD: [], BWD: []},
                ready_out={FWD: [], BWD: []},
            )
            for p in range(G)
        ]
        hops_dir = lambda s, t, d: (t - s) % G if d == FWD else (s - t) % G
        route = None
        if ring and DIRECT_RING_SENDS and G >= 3:
            route = {
                (s, t): _first_hop_dir(mesh, grp[s], grp[t], nxt[s], prv[s])
                for s in range(G)
                for t in range(G)
                if s != t
            }
            if None in route.values():
                route = None
        if route is not None:
            for t in range(G):
                for d in (FWD, BWD):
                    srcs = [s for s in range(G) if s != t and route[(s, t)] == d]
                    info[t]["entries"][d] = sorted(srcs, key=lambda s: hops_dir(s, t, d))
            for s in range(G):
                info[s]["direct"] = True
                for d in (FWD, BWD):
                    dsts = sorted(
                        (t for t in range(G) if t != s and route[(s, t)] == d), key=lambda t: hops_dir(s, t, d)
                    )
                    info[s]["sends"][d] = [(s, t, info[t]["entries"][d].index(s), hops_dir(s, t, d), 0) for t in dsts]
        else:
            for s in range(G):
                fwd, bwd = _schedule(s, G, ring)
                for d, lst, peer in ((FWD, fwd, nxt[s]), (BWD, bwd, prv[s])):
                    if peer is None:
                        continue
                    t = grp.index(peer)
                    info[s]["sends"][d] = [(q, t, i, 1, 0 if i == 0 else i) for i, q in enumerate(lst)]
                    info[t]["entries"][d] = list(lst)
        # ready fence: one packet per (sending port, destination chip)
        for s in range(G):
            for d in (FWD, BWD):
                for t in sorted({it[1] for it in info[s]["sends"][d]}):
                    back = route[(t, s)] if route is not None else _OPP[d]
                    info[t]["ready_out"][back].append((s, d, hops_dir(t, s, back)))
        for s in range(G):
            for d in (FWD, BWD):
                info[s]["sends"][d] = [(q, grp[t], i, h, wt) for q, t, i, h, wt in info[s]["sends"][d]]
                info[s]["ready_out"][d] = [(grp[t], dd, h) for t, dd, h in info[s]["ready_out"][d]]
            chips[grp[s]] = info[s]
    return chips


# ----------------------------------------------------------------------------------------------------------------------
# Local plan
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class Group:
    start_row: int
    rows: int
    cores: list  # logical CoreCoord, root first
    slices: list  # (w0, V) per core
    owners: int = 1  # roots of the group: cores[:owners] (R3-RS: > 1)
    links: list = field(default_factory=list)  # per owner: its link
    offsets: list = field(default_factory=list)  # per owner: byte offset of its rows in its link's segment, per wave


@dataclass
class Plan:
    G: int
    ring: bool
    L_act: int
    ports: dict  # (d, l) -> logical CoreCoord
    regime: str  # "R1" / "R3"
    Wt: int
    S: int
    C: int
    nch: int
    WS: int
    G_w: int
    groups: list
    R: int
    num_waves: int
    xd: int
    gd: int
    idp: int
    slot_depth: int  # cb_stats_slots regions of G * Q pages (2 under the look-ahead)
    seg_bytes: dict  # (w, l) -> bytes
    seg_stride: int
    landing_bytes: int
    x_res: bool = False  # cb_x_tiles backed on the input shard (zero-copy)
    out_res: bool = False  # cb_output backed on the output shard (zero-copy; compute packs into it)
    gamma_resident: bool = True  # γ: WS-page window read once (True) or GAMMA_STREAM_DEPTH C-page chunks per block
    owners: int = 1  # R3-RS owners per row-group (1 = single root)
    pipe: bool = False  # WAVE_PIPELINE look-ahead across waves (see _depths)


def _chunking(S, x_res=False, cap=W_CHUNK_TILES):
    """(C, nch, WS). Interleaved: C = ceil(S / ceil(S / cap)), WS = nch·C (the x ring's row stride; pad tiles zeroed).
    Shard-backed: WS = S (the shard's own row stride) and C = the largest divisor of S <= cap, so every chunk is
    exactly C wide: every CB quantum is then C pages and no push ever straddles a ring end (pack / unpack addresses
    do not wrap inside a block)."""
    if x_res:
        C = max(d for d in range(1, min(S, cap) + 1) if S % d == 0)
        return C, S // C, S
    nch = math.ceil(S / cap)
    C = math.ceil(S / nch)
    return C, nch, nch * C


def _tile_rows(shape):
    return math.prod(shape[:-2]) * math.ceil(shape[-2] / TILE_DIM)


def pipe_options(num_waves):
    """Look-ahead settings the solve tries, preferred first (WAVE_PIPELINE, and only when there is a next wave)."""
    return [True, False] if (WAVE_PIPELINE and num_waves > 1) else [False]


def _depths(num_waves, pipe=False):
    """(x_depth, gather_depth, inv_depth, slot_depth). x is prefetched a block ahead (the reader's DRAM reads overlap
    compute). Without the look-ahead cb_gather / cb_stats_slots need ONE block: a member's partials of block b+1 exist
    only after its compute consumed inv(b), which every owner sent only after its compute had consumed gather(b) — so a
    second slot could never fill early. With it (pipe), block b+1's partials / own stats are produced while block b's
    still wait for the exchange: two slots each (never three: block b+2's need inv(b)). cb_inv_rms stays 1.
    pipe: the interleaved x ring holds every wave (x_depth = num_waves): compute squares block w+1 while block w is still
    at the front, at a fixed page offset, which a 2-block ring could not express once it wraps (indexed unpack reads do
    not wrap). The solve picks pipe only where that footprint fits (wave counts the plan chose for overlap, R3_MIN_WAVES /
    BLOCK_ROWS_CAP), never for waves L1 forced."""
    if pipe:
        return num_waves, 2, 1, 2
    return min(MAX_PIPELINE_DEPTH, num_waves), 1, 1, 1


def footprint(
    R,
    num_waves,
    *,
    WS,
    C,
    G,
    G_w,
    r3,
    has_gamma,
    T_in,
    T_g,
    x_res=False,
    out_res=False,
    gamma_resident=True,
    owners=1,
    pipe=False,
    T_dest=None,
):
    """Compute-core CB bytes (l1_ledger.md -> Total per-core footprint). Shard-backed CBs (x_res / out_res) own no L1
    of their own: they alias the input / output shard, which is already allocated (and already out of the budget)."""
    xd, gd, idp, slot_depth = _depths(num_waves, pipe)
    T32 = ttnn.tile_size(STATS_FORMAT)
    T_dest = T32 if T_dest is None else T_dest  # DEST-following pages (sumsq, x_gamma): dest_format()
    Q = math.ceil(R / owners)  # rows a root owns per block: the root-only CBs scale with Q, every core's with R
    per_row = (
        (0 if x_res else xd * WS * T_in)
        + T_dest  # sumsq
        + T32 * (idp + (1 if r3 else 0))  # inv_rms, [partial_col]
        + (C * T_dest if has_gamma else 0)  # x_gamma
    )
    per_owned_row = T32 * (
        1 + slot_depth * G + (1 + gd * G_w if r3 else 0)
    )  # device_partial, stats slots, [inv_root, gather]
    gamma_pages = WS if gamma_resident else GAMMA_STREAM_DEPTH * C
    fixed = (
        ttnn.tile_size(SCALER_FORMAT)
        + (gamma_pages * T_g if has_gamma else 0)
        + (0 if out_res else STREAM_DEPTH * C * T_in)
    )
    return R * per_row + Q * per_owned_row + fixed


def rs_owners(G_w, R):
    """R3-RS owners of an R-row block (see R3_RS_OWNERS)."""
    cap = G_w if R3_RS_OWNERS is None else max(1, min(G_w, R3_RS_OWNERS))
    return max(d for d in range(1, min(cap, R) + 1) if R % d == 0)


def owner_rows(R_b, owners, k):
    """Rows owner k sums in a block of R_b rows: [k*q, k*q + n), q = ceil(R_b / owners) (mirrored by the kernels)."""
    if R_b == 0:
        return 0
    q = math.ceil(R_b / owners)
    return max(0, min(q, R_b - k * q))


def _port_cores(grid, L_act, busy=frozenset()):
    """2·L_act port cores, the same logical cores on every chip. Preferred: logical row 0, centred. When a compute
    core (a sharded input's shard grid, `busy`) sits there: the last free cores of the grid in reverse row-major order
    (shard grids fill from (0, 0)). None when fewer than 2·L_act cores are free."""
    n = 2 * L_act
    x0 = max(0, grid.x // 2 - L_act) if PORT_X0 is None else max(0, min(PORT_X0, grid.x - n))
    pref = [(x0 + i, 0) for i in range(n)]
    if x0 + n <= grid.x and not busy.intersection(pref):
        picked = pref
    else:
        free = [(x, y) for y in reversed(range(grid.y)) for x in reversed(range(grid.x)) if (x, y) not in busy]
        if len(free) < n:
            return None
        picked = sorted(free[:n], key=lambda c: (c[1], c[0]))
    cores = [ttnn.CoreCoord(x, y) for x, y in picked]
    return {(d, l): cores[2 * l + d] for l in range(L_act) for d in (FWD, BWD)}


def same_placement(out_mc, in_mc) -> bool:
    """The output carries the input's exact shard placement (layout, buffer type, grid, shard shape, orientation): the
    output shard of every core is then that core's own, and compute packs straight into it (zero-copy)."""
    if out_mc.memory_layout != in_mc.memory_layout or in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return False
    a, b = out_mc.shard_spec, in_mc.shard_spec
    return (
        out_mc.buffer_type == in_mc.buffer_type
        and a is not None
        and b is not None
        and list(a.shape) == list(b.shape)
        and a.grid == b.grid
        and a.orientation == b.orientation
    )


def _shard_layout(input_tensor, M_t, Wt):
    """A sharded input: the shard spec IS the work split (regime, groups, G_w, S). Cores whose shard holds no valid data
    get no work. HEIGHT -> R1-X (one group per shard core: whole local rows, the partial stays on the core).
    WIDTH -> R3-X with one row-group: every core holds all M_t tile-rows of its S-tile W-slice (shard order)."""
    mc = input_tensor.memory_config()
    spec = mc.shard_spec
    if spec is None or mc.buffer_type != ttnn.BufferType.L1:
        raise NotImplementedError("distributed_rms_norm: sharded inputs must carry an L1 shard spec")
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    if shard_h % TILE_DIM or shard_w % TILE_DIM:
        raise NotImplementedError("distributed_rms_norm: TILE shards must be tile multiples")
    Hs, S = shard_h // TILE_DIM, shard_w // TILE_DIM
    n = spec.grid.num_cores()
    cores = list(ttnn.corerange_to_cores(spec.grid, n, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))
    if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if S < Wt or n * Hs < M_t:
            raise NotImplementedError("distributed_rms_norm: height shard spec does not cover the tensor")
        groups = [
            Group(i * Hs, min(Hs, M_t - i * Hs), [core], [(0, Wt)]) for i, core in enumerate(cores) if i * Hs < M_t
        ]
        return "R1", groups, 1, S
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if Hs < M_t or n * S < Wt:
            raise NotImplementedError("distributed_rms_norm: width shard spec does not cover the tensor")
        used = math.ceil(Wt / S)
        slices = [(j * S, min(S, Wt - j * S)) for j in range(used)]
        return ("R3" if used >= 2 else "R1"), [Group(0, M_t, cores[:used], slices)], used, S
    raise NotImplementedError(f"distributed_rms_norm: {mc.memory_layout} is not supported")


def _r1_groups(cores, M_t, Wt):
    n = min(len(cores), M_t)
    base, rem = divmod(M_t, n)
    groups, start = [], 0
    for i in range(n):
        rows = base + (1 if i < rem else 0)
        groups.append(Group(start, rows, [cores[i]], [(0, Wt)]))
        start += rows
    return groups


def _r3_layout(n_cores, M_t, Wt):
    groups = math.ceil(M_t / R3_ROWS_PER_GROUP)
    G_w = min(R3_MAX_GROUP_CORES, n_cores // groups, Wt // R3_MIN_SLICE_TILES)
    if G_w < 2:
        return None
    S = math.ceil(Wt / G_w)
    G_w = math.ceil(Wt / S)
    return groups, G_w, S


def _segments(free, n, along_x):
    """Hole-free line segments of n cores of the set `free` ((x, y) tuples): each row (along_x) or column carved
    left-to-right / top-to-bottom into whole runs of n consecutive free cores."""
    out = []
    lines = sorted({c[1] if along_x else c[0] for c in free})
    for v in lines:
        line = sorted(c for c in free if (c[1] if along_x else c[0]) == v)
        run = []
        for c in line:
            pos = c[0] if along_x else c[1]
            if run and pos != (run[-1][0] if along_x else run[-1][1]) + 1:
                run = []
            run.append(c)
            if len(run) == n:
                out.append(run)
                run = []
    return out


def _member_rects(cores):
    """Logical rectangles covering `cores` (a root's members): their bounding box when it is hole-free, else their
    maximal x-runs per row. Each is ((x0, y0), (x1, y1)) with x0 <= x1, y0 <= y1."""
    if not cores:
        return []
    xs, ys = [c.x for c in cores], [c.y for c in cores]
    if (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1) == len(cores):
        return [((min(xs), min(ys)), (max(xs), max(ys)))]
    out = []
    for y in sorted(set(ys)):
        row = sorted(c.x for c in cores if c.y == y)
        s = row[0]
        for i in range(1, len(row) + 1):
            if i == len(row) or row[i] != row[i - 1] + 1:
                out.append(((s, y), (row[i - 1], y)))
                if i < len(row):
                    s = row[i]
    return out


def _fwd_rect_args(mesh, members, noc):
    """The reader RT words of a FWD_INV root's member multicast: [n_rect] + (x_start, y_start, x_end, y_end,
    num_dests) per rectangle, in virtual coordinates ordered for the root reader's NoC (NoC1 multicasts start at the
    high corner, as kernel_lib's noc_ordered_bounds)."""
    words = []
    rects = _member_rects(members)
    for (x0, y0), (x1, y1) in rects:
        lo = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
        hi = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
        corners = [lo.x, lo.y, hi.x, hi.y] if noc == ttnn.NOC.NOC_0 else [hi.x, hi.y, lo.x, lo.y]
        words += corners + [(x1 - x0 + 1) * (y1 - y0 + 1)]
    return [len(rects)] + words


def _rect_groups(cores, n_groups, G_w):
    """n_groups single-rectangle row-groups of G_w cores (RECT_GROUPS: one slot-forward rectangle per group, the root
    at one end): row segments first, then column segments of what is left; None when they do not cover n_groups."""
    free = {(c.x, c.y) for c in cores}
    rows = _segments(free, G_w, True)
    left = free - {c for seg in rows for c in seg}
    segs = rows + _segments(left, G_w, False)
    if len(segs) < n_groups:
        return None
    return [[ttnn.CoreCoord(x, y) for x, y in seg] for seg in segs[:n_groups]]


def _r3_groups(cores, M_t, Wt, n_groups, G_w, S):
    rect = _rect_groups(cores, n_groups, G_w) if RECT_GROUPS else None
    out = []
    for gi in range(n_groups):
        start = gi * R3_ROWS_PER_GROUP
        rows = min(R3_ROWS_PER_GROUP, M_t - start)
        cs = rect[gi] if rect is not None else cores[gi * G_w : (gi + 1) * G_w]
        slices = [(j * S, min(S, Wt - j * S)) for j in range(G_w)]
        out.append(Group(start, rows, cs, slices))
    return out


def _l1_blocks(mesh):
    """L1 allocator blocks (address relative to the allocator base, size, allocated), ascending."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    return sorted((int(b["address"]), int(b["size"]), b["allocated"] == "yes") for b in view.block_table)


def _l1_budget(mesh):
    """Bytes the program's CBs may use. CBs grow up from the allocator base and must end below the LOWEST occupied L1
    address (the allocator is bank-uniform, so any L1 buffer anywhere caps every core), i.e. the free block at the
    bottom of L1 — not the largest free block: a hole above a low allocation (a freed shard above a buffer created
    under it) is unusable by CBs."""
    blocks = _l1_blocks(mesh)
    if not blocks:
        return ttnn.get_memory_view(mesh, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    addr, size, allocated = blocks[0]
    return size if addr == 0 and not allocated else 0


_PERSIST = {}  # (id(mesh), port cores) -> {"mesh", "crs", "sems": (ready, arrival_0, ...), "scratch": {bytes: tensor}}


def _n_arrival_counters(mesh):
    """Arrival counters per port core: one per entry that can land on one port in one wave (<= G - 1 for any group of
    this mesh: a Linear line end receives G - 1 relayed entries, a direct Ring port at most G - 1 sources)."""
    return max(1, max(tuple(mesh.shape)) - 1)


def _new_sems(mesh, port_crs):
    """(ready, arrival_0, ..., arrival_{n-1}): global semaphores on the port cores, zero-initialized."""
    return tuple(ttnn.create_global_semaphore(mesh, port_crs, 0) for _ in range(1 + _n_arrival_counters(mesh)))


def _new_scratch(mesh, port_crs, size):
    words = size // 4
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(port_crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([port_crs.num_cores(), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc
    )


def _settle_persistent(mesh):
    """Move this mesh's persistent buffers UP when they cap the CB region and a higher free block can hold them.

    They are allocated on first use, top-down under whatever L1 tensors are live then — under a sharded input's shards,
    say — and never freed, so without this they would pin the lowest occupied L1 address (= the end of every later
    program's CB region, `_l1_budget`) for the process even after those shards are gone (measured: 1.46 MB -> 0.42 MB
    of CB space after one 512 KB-shard call). New buffers are created first, while the old ones are still live, so
    they land in the highest free block (strictly above the old ones); then the old ones are freed. Monotone (buffers
    only ever move up), so a steady state never churns. Runs before the plan, after a device sync (no in-flight call
    uses the old addresses); a trace captured before a move must be re-captured (its RT args hold the old addresses)."""
    base = int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1))
    for key, entry in list(_PERSIST.items()):
        if entry["mesh"] is not mesh:
            continue
        blocks = _l1_blocks(mesh)
        lowest_occupied = next((a for a, _, allocated in blocks if allocated), None)
        addrs = [int(ttnn.get_global_semaphore_address(sm)) - base for sm in entry["sems"]]
        addrs += [int(t.buffer_address()) - base for t in entry["scratch"].values()]
        need = max(entry["scratch"], default=64)
        mine = min(addrs)
        if lowest_occupied is None or mine != lowest_occupied:
            continue  # not what caps the CB region
        if not any(not allocated and a > mine and size >= need for a, size, allocated in blocks):
            continue  # nowhere higher to go
        ttnn.synchronize_device(mesh)
        crs = entry["crs"]
        moved = {"mesh": mesh, "crs": crs, "sems": _new_sems(mesh, crs), "scratch": {}}
        for size in sorted(entry["scratch"], reverse=True):
            moved["scratch"][size] = _new_scratch(mesh, crs, size)
        for t in entry["scratch"].values():
            t.deallocate()
        entry["scratch"].clear()
        entry["sems"] = None  # drop the old semaphores (their buffers are freed with the objects)
        _PERSIST[key] = moved
        ttnn.synchronize_device(mesh)  # the new counters are zero on every chip before any chip's next call


def _persistent_key(mesh, ports):
    return (id(mesh), tuple(sorted((c.x, c.y) for c in ports.values())))


def _landing_bucket(nbytes: int) -> int:
    """Allocation size for a landing scratch of `nbytes`: the next power of two >= LANDING_MIN_BYTES.

    The scratch is persistent (never freed: traced programs keep its address), and a sharded L1 buffer reserves its
    range on every L1 bank, so one scratch per distinct plan size would grow L1 use without bound across shapes and
    shrink every later plan's L1 budget. Buckets bound the set to ~log2(max / LANDING_MIN_BYTES) buffers, < 2x the
    largest, and any cached bucket >= the need is reused."""
    return max(LANDING_MIN_BYTES, 1 << (max(1, nbytes) - 1).bit_length())


def _live_scratch_sizes(key) -> tuple:
    """Bucket sizes of the still-allocated cached scratches for `key` (ints only: no tensor is held in a caller's
    frame, so nothing on an error path can see / free the persistent buffers). Drops externally freed ones."""
    scratch = _PERSIST.get(key, {}).get("scratch", {})
    for size in [sz for sz in scratch if not scratch[sz].is_allocated()]:
        del scratch[size]
    return tuple(sorted(scratch))


def make_plan(
    mesh, input_tensor, gamma, cluster_axis, topology, num_links, out_memory_config=None, fp32_dest_acc_en=True
):
    """out_memory_config: the output's placement (None = the input's own placement). fp32_dest_acc_en: sizes the
    DEST-following CBs (dest_format)."""
    shape = list(input_tensor.shape)
    G = tuple(mesh.shape)[cluster_axis]
    ring = topology == Topology.Ring and G > 2
    M_t = _tile_rows(shape)
    Wt = math.ceil(shape[-1] / TILE_DIM)
    has_gamma = gamma is not None
    T_in = ttnn.tile_size(input_tensor.dtype)
    T_g = ttnn.tile_size(gamma.dtype) if has_gamma else 0
    T_dest = ttnn.tile_size(dest_format(fp32_dest_acc_en))
    grid = mesh.compute_with_storage_grid_size()
    _settle_persistent(mesh)
    budget_now = _l1_budget(mesh)
    in_mc = input_tensor.memory_config()
    x_res = in_mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    out_res = x_res and (out_memory_config is None or same_placement(out_memory_config, in_mc))

    def _owners_of(regime, G_w, R):
        # R3-RS for WIDTH_SHARDED row-groups (all rows in one group); a single root everywhere else
        return rs_owners(G_w, R) if (x_res and regime == "R3") else 1

    for L_act in range(num_links, 0, -1):
        if x_res:
            # The shard spec fixes the compute cores; the ports take free cores outside the shard grid.
            shard_regime, shard_groups, shard_G_w, shard_S = _shard_layout(input_tensor, M_t, Wt)
            busy = frozenset((c.x, c.y) for g in shard_groups for c in g.cores)
            ports = _port_cores(grid, L_act, busy)
            if ports is None:
                if L_act > 1:
                    continue
                raise NotImplementedError(
                    "distributed_rms_norm: the shard grid leaves fewer than 2 free cores for the Fabric port cores"
                )
        else:
            ports = _port_cores(grid, L_act)
        port_set = {(c.x, c.y) for c in ports.values()}
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in port_set]
        cached_sizes = _live_scratch_sizes(_persistent_key(mesh, ports))

        def solve(regime, G_w, S, groups, L):
            """(R, (chunk cap, γ resident), pipe): the first l1_fallbacks() layout with an R >= 1, and its largest R
            whose footprint (+ the landing scratch, when not yet allocated) fits, with the look-ahead if it fits at
            that R; (0, None, False) if none. A shard-backed block defaults to the whole shard (max_rows), clamped
            only by this solve."""
            for layout in l1_fallbacks():
                R, pipe = solve_rows(regime, G_w, S, groups, L, *layout)
                if R:
                    return R, layout, pipe
            return 0, None, False

        def solve_rows(regime, G_w, S, groups, L, chunk_cap, gamma_resident):
            C, nch, WS = _chunking(S, x_res, chunk_cap)
            max_rows = max(g.rows for g in groups)
            waves = R3_MIN_WAVES if (regime == "R3" and not x_res) else 1
            cap = math.ceil(max_rows / max(1, waves))
            cap = cap if BLOCK_ROWS_CAP is None else max(1, min(cap, BLOCK_ROWS_CAP))

            def fits(R, pipe):
                nw = math.ceil(max_rows / R)
                owners = _owners_of(regime, G_w, R)
                f = footprint(
                    R,
                    nw,
                    WS=WS,
                    C=C,
                    G=G,
                    G_w=G_w,
                    r3=regime == "R3",
                    has_gamma=has_gamma,
                    T_in=T_in,
                    T_g=T_g,
                    x_res=x_res,
                    out_res=out_res,
                    gamma_resident=gamma_resident,
                    owners=owners,
                    pipe=pipe,
                    T_dest=T_dest,
                )
                seg = max(1, _seg_rows_max(groups, owners, R, nw, L))
                land = nw * G * _align_up(seg * STATS_ROW_BYTES, SEG_ALIGN)
                return f + (0 if any(sz >= land for sz in cached_sizes) else _landing_bucket(land)) <= budget_now

            def rs_ok(R):
                # R3-RS: a block size whose owner count is near-full (a prime R > G_w would collapse to one root)
                if not (x_res and regime == "R3"):
                    return True
                return 2 * _owners_of(regime, G_w, R) > min(G_w, R)

            for R in range(cap, 0, -1):
                for pipe in pipe_options(math.ceil(max_rows / R)):
                    if rs_ok(R) and fits(R, pipe):
                        # Balanced blocks: the same wave count with the smallest even R (39 + 25 rows -> 32 + 32), so
                        # no block carries the longer per-block critical path. A no-op when everything fits in one block.
                        balanced = math.ceil(max_rows / math.ceil(max_rows / R))
                        if balanced < R and rs_ok(balanced) and fits(balanced, pipe):
                            return balanced, pipe
                        return R, pipe
            return 0, False

        if x_res:
            fit, _, _ = solve(shard_regime, shard_G_w, shard_S, shard_groups, L_act)
            if fit and _n_roots(shard_groups, _owners_of(shard_regime, shard_G_w, fit)) < L_act:
                continue  # fewer roots than links (WIDTH, single root): retry with fewer port cores
            if not fit:
                if L_act > 1:
                    continue  # fewer links -> fewer port cores and a smaller landing
                raise NotImplementedError(
                    "distributed_rms_norm: one resident tile-row (block_rows = 1) does not fit L1 next to the shards"
                )
            regime, groups, G_w, S = shard_regime, shard_groups, shard_G_w, shard_S
            break

        r1_cores = min(len(cores), M_t)
        r3 = _r3_layout(len(cores), M_t, Wt)
        r1_groups = _r1_groups(cores, M_t, Wt)
        n_r1 = len(r1_groups)
        r1_fit, _, _ = solve("R1", 1, Wt, r1_groups, min(L_act, n_r1))
        choice = None
        if r3 is not None:
            n_groups, G_w, S = r3
            r3_groups = _r3_groups(cores, M_t, Wt, n_groups, G_w, S)
            r3_fit, _, _ = solve("R3", G_w, S, r3_groups, min(L_act, n_groups))
            occupancy = G_w >= R3_MIN_GROUP_CORES and G_w * n_groups >= R3_OCCUPANCY_GAIN * r1_cores
            if r3_fit and (occupancy or not r1_fit):
                choice = ("R3", r3_groups, G_w, S)
        if choice is None and r1_fit:
            choice = ("R1", r1_groups, 1, Wt)
        if choice is None:
            raise NotImplementedError(
                "distributed_rms_norm: one resident tile-row (block_rows = 1) does not fit L1 (footprint(R=1) > budget)"
            )
        regime, groups, G_w, S = choice
        if len(groups) < L_act:
            continue  # fewer row-groups than links: retry with fewer port cores
        break

    max_rows = max(g.rows for g in groups)
    R, (chunk_cap, gamma_resident), pipe = solve(regime, G_w, S, groups, L_act)
    assert R >= 1
    owners = _owners_of(regime, G_w, R)
    C, nch, WS = _chunking(S, x_res, chunk_cap)
    num_waves = math.ceil(max_rows / R)
    xd, gd, idp, slot_depth = _depths(num_waves, pipe)
    # link split: contiguous runs of roots (row-group roots; R3-RS: every owner), balanced by root count
    roots = _roots(groups, owners)
    for g in groups:
        g.owners = owners
        g.links = [0] * owners
        g.offsets = [[] for _ in range(owners)]
    for (g, k), l in zip(roots, _root_links(len(roots), L_act)):
        g.links[k] = l
    seg_bytes = {}
    for l in range(L_act):
        for w in range(num_waves):
            off = 0
            for g, k in roots:
                if g.links[k] == l:
                    g.offsets[k].append(off)
                    off += owner_rows(_wave_rows(g.rows, R, w), owners, k) * STATS_ROW_BYTES
            seg_bytes[(w, l)] = off
    seg_stride = _align_up(max(1, max(seg_bytes.values())), SEG_ALIGN)
    return Plan(
        G=G,
        ring=ring,
        L_act=L_act,
        ports=ports,
        regime=regime,
        Wt=Wt,
        S=S,
        C=C,
        nch=nch,
        WS=WS,
        G_w=G_w,
        groups=groups,
        R=R,
        num_waves=num_waves,
        xd=xd,
        gd=gd,
        idp=idp,
        slot_depth=slot_depth,
        seg_bytes=seg_bytes,
        seg_stride=seg_stride,
        landing_bytes=num_waves * G * seg_stride,
        x_res=x_res,
        out_res=out_res,
        gamma_resident=gamma_resident or not has_gamma,
        owners=owners,
        pipe=pipe,
    )


def _wave_rows(rows, R, w):
    return max(0, min(R, rows - w * R))


def _roots(groups, owners):
    """(group, owner index) of every root, in group order (owners = 1: one root per row-group)."""
    return [(g, k) for g in groups for k in range(owners)]


def _n_roots(groups, owners):
    return len(groups) * owners


def _root_links(n, L):
    """Link of each of n roots: contiguous runs, balanced by count."""
    return [l for l in range(L) for _ in range(round(l * n / L), round((l + 1) * n / L))]


def _seg_rows_max(groups, owners, R, nw, L):
    """Max rows of one link's segment over the links and waves."""
    roots = _roots(groups, owners)
    links = _root_links(len(roots), L)
    return max(
        sum(owner_rows(_wave_rows(g.rows, R, w), owners, k) for (g, k), lk in zip(roots, links) if lk == l)
        for l in range(L)
        for w in range(nw)
    )


def _persistent(mesh, plan):
    """Global semaphores (ready, arrival counters) and a landing scratch >= plan.landing_bytes for this port-core set;
    created once, never freed (bucketed: see _landing_bucket)."""
    key = _persistent_key(mesh, plan.ports)
    entry = _PERSIST.get(key)
    port_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in plan.ports.values()])
    if entry is None:
        entry = {"mesh": mesh, "crs": port_crs, "sems": _new_sems(mesh, port_crs), "scratch": {}}
        _PERSIST[key] = entry
        new = True
    else:
        new = False
    fits = [sz for sz in _live_scratch_sizes(key) if sz >= plan.landing_bytes]
    size = fits[0] if fits else _landing_bucket(plan.landing_bytes)
    if size not in entry["scratch"]:
        entry["scratch"][size] = _new_scratch(mesh, port_crs, size)
    if new:
        # every chip's counters must be zero before any chip's first call can increment a neighbour's
        ttnn.synchronize_device(mesh)
    ready, *arrivals = (int(ttnn.get_global_semaphore_address(s)) for s in entry["sems"])
    landing = int(entry["scratch"][size].buffer_address())
    # A real L1 allocation never sits at address 0. Address 0 means the resources were created while dispatch was
    # blocked (e.g. a no-dispatch program-collection pass: mock allocations, the zero-init never written). Use them
    # for this (non-dispatched) build only and never cache them: a later real call must get real ones.
    if ready == 0 or 0 in arrivals:
        _PERSIST.pop(key, None)
    elif landing == 0:
        entry["scratch"].pop(size, None)
    return arrivals, ready, landing


# ----------------------------------------------------------------------------------------------------------------------
# Mesh program
# ----------------------------------------------------------------------------------------------------------------------


def _cb(index, fmt, page, pages, crs):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _kernel_config(cfg):
    out = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        fp32_dest_acc_en=cfg.fp32_dest_acc_en,
        math_approx_mode=cfg.math_approx_mode,
    )
    out.dst_full_sync_en = cfg.dst_full_sync_en
    modes = [ttnn.UnpackToDestMode.Default] * 64  # one entry per circular buffer
    # Exact fp32 cross-device sum: the slots unpack straight into a 32-bit DEST. With 16-bit DEST the unpacker would
    # write 32-bit words into a DEST the math / pack threads read as 16-bit (garbage / inf, measured): Default, the
    # slots enter DEST through srcA (rounded to the DEST width) and AddCol0 sums them in the SFPU's fp32 lanes.
    if cfg.fp32_dest_acc_en:
        modes[CB_STATS_SLOTS] = ttnn.UnpackToDestMode.UnpackToDestFp32
    out.unpack_to_dest_mode = modes
    return out


def gamma_row0_prefix_bytes(gamma) -> int:
    """Bytes the writer reads per γ tile. TILE: [0, n) of the tile, row 0 of faces 0 and 1 (bf16 544 B, fp32 1088 B).
    ROW_MAJOR sticks: the whole 32-element stick (bf16 64 B, fp32 128 B = its aligned DRAM page); the writer moves its
    second half to face 1 row 0 in L1."""
    es = ttnn.tile_size(gamma.dtype) // (TILE_DIM * TILE_DIM)
    if gamma.layout == ttnn.ROW_MAJOR_LAYOUT:
        return TILE_DIM * es
    return FACE_DIM * FACE_DIM * es + FACE_DIM * es


def w_last_valid(input_tensor) -> int:
    """Valid columns of the last tile of a row (32 when W_local is tile-aligned); the reader zeroes the rest."""
    return (int(input_tensor.shape[-1]) - 1) % TILE_DIM + 1


def noc_swap_cores(plan, grid_x):
    """Compute cores whose reader / writer swap NoCs (DRAM NoC balance, see NOC_SWAP_COLUMNS): the NOC_SWAP_COLUMNS
    westmost logical columns plus the east-most one when it holds >= NOC_SWAP_EAST_MIN_CORES compute cores, minus the
    FWD_INV roots (multicast senders). Nothing swaps when neither x nor the output crosses DRAM."""
    if plan.x_res and plan.out_res:
        return []
    cores = [c for g in plan.groups for c in g.cores]
    east = [c for c in cores if c.x == grid_x - 1]
    cols = {c.x for c in cores if c.x < NOC_SWAP_COLUMNS}
    if len(east) >= NOC_SWAP_EAST_MIN_CORES:
        cols.add(grid_x - 1)
    senders = (
        {(g.cores[0].x, g.cores[0].y) for g in plan.groups} if (plan.regime == "R3" and plan.owners == 1) else set()
    )
    return [c for c in cores if c.x in cols and (c.x, c.y) not in senders]


def _split_noc_swapped(kd, cores, swapped, processor, noc, swapped_noc):
    """[descriptors]: `kd` (on `processor` / `noc`) restricted to the unswapped cores, plus (when `swapped` is
    non-empty) a same-source copy on the swapped cores on `swapped_noc` (same processor, same CT / RT args)."""
    if not swapped:
        return [kd]
    sw = {(c.x, c.y) for c in swapped}
    normal = [c for c in cores if (c.x, c.y) not in sw]

    def sub(cs, n):
        rt = ttnn.RuntimeArgs()
        for c in cs:
            rt[c.x][c.y] = list(kd.runtime_args[c.x][c.y])
        return ttnn.KernelDescriptor(
            kernel_source=kd.kernel_source,
            core_ranges=_crs(cs),
            compile_time_args=list(kd.compile_time_args),
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=n),
        )

    out = [sub(swapped, swapped_noc)]
    if normal:  # every core swapped: no empty-range descriptor (it would carry no CB formats)
        out.insert(0, sub(normal, noc))
    return out


def create_mesh_program_descriptor(
    input_tensor, output_tensor, *, gamma, epsilon, cluster_axis, topology, num_links, compute_kernel_config
):
    mesh = input_tensor.device()
    plan = make_plan(
        mesh,
        input_tensor,
        gamma,
        cluster_axis,
        topology,
        num_links,
        out_memory_config=output_tensor.memory_config(),
        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
    )
    arrival_addrs, ready_addr, landing_addr = _persistent(mesh, plan)
    G, L = plan.G, plan.L_act
    has_gamma = gamma is not None
    r3 = plan.regime == "R3"
    fwd_inv = r3 and plan.owners == 1  # the kernels' FWD_INV: the root forwards the slots, every member finalizes
    assert not fwd_inv or G <= 24, "SEM_SLOT_FWD packs one bit per slot below the block tag"
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    W_global = G * int(input_tensor.shape[-1])

    assert DM_NOCS[0] != DM_NOCS[1], "DM_NOCS: the reader and writer must use different NoCs"
    compute_cores = [c for g in plan.groups for c in g.cores]
    compute_crs = _crs(compute_cores)
    swapped = noc_swap_cores(plan, mesh.compute_with_storage_grid_size().x)
    port_list = [plan.ports[(d, l)] for l in range(L) for d in (FWD, BWD)]
    port_crs = _crs(port_list)
    all_crs = _crs(compute_cores + port_list)
    T_in = ttnn.tile_size(input_tensor.dtype)
    T_out = ttnn.tile_size(output_tensor.dtype)
    T32 = ttnn.tile_size(STATS_FORMAT)
    DEST_FORMAT = dest_format(bool(compute_kernel_config.fp32_dest_acc_en))
    T_dest = ttnn.tile_size(DEST_FORMAT)
    R, WS, C = plan.R, plan.WS, plan.C
    Q = R // plan.owners  # rows a root owns per full block (owners divides R)

    # ---------- CBs (compute cores; port cores carry none) ----------
    cbs = [
        # Zero-copy: compute reads x straight out of the input shard / packs into the output shard.
        (
            ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input_tensor)
            if plan.x_res
            else _cb(CB_X_TILES, input_tensor.dtype, T_in, plan.xd * R * WS, compute_crs)
        ),
        _cb(CB_REDUCE_SCALER, SCALER_FORMAT, ttnn.tile_size(SCALER_FORMAT), 1, compute_crs),
        _cb(CB_SUMSQ_PARTIAL, DEST_FORMAT, T_dest, R, compute_crs),
        _cb(CB_DEVICE_PARTIAL, STATS_FORMAT, T32, Q, compute_crs),
        _cb(CB_STATS_SLOTS, STATS_FORMAT, T32, plan.slot_depth * G * Q, compute_crs),
        _cb(CB_INV_RMS, STATS_FORMAT, T32, plan.idp * R, compute_crs),
        (
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor)
            if plan.out_res
            else _cb(CB_OUTPUT, output_tensor.dtype, T_out, STREAM_DEPTH * C, compute_crs)
        ),
    ]
    if r3:
        cbs += [
            _cb(CB_PARTIAL_COL, STATS_FORMAT, T32, R, compute_crs),
            _cb(CB_GATHER, STATS_FORMAT, T32, plan.gd * plan.G_w * Q, compute_crs),
            _cb(CB_INV_ROOT, STATS_FORMAT, T32, Q, compute_crs),
        ]
    if has_gamma:
        cbs += [
            _cb(
                CB_GAMMA_TILES,
                gamma.dtype,
                ttnn.tile_size(gamma.dtype),
                WS if plan.gamma_resident else GAMMA_STREAM_DEPTH * C,
                compute_crs,
            ),
            _cb(CB_X_GAMMA, DEST_FORMAT, T_dest, R * C, compute_crs),
        ]

    # ---------- CT args ----------
    common_ct = [
        R,
        plan.num_waves,
        WS,
        C,
        plan.nch,
        plan.Wt,
        G,
        plan.G_w,
        int(r3),
        int(has_gamma),
        plan.xd,
        plan.gd,
        SEM_GATHER,
        SEM_INV,
        SEM_STATS_LANDED,
        SEM_OWN_STATS,
        gamma_row0_prefix_bytes(gamma) if has_gamma else 0,
        SEM_OWN_COUNTED,
        int(plan.x_res),
        int(plan.out_res),
        int(plan.gamma_resident),
        plan.owners,
        w_last_valid(input_tensor),
        int(has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT),
        plan.slot_depth,  # cb_stats_slots regions; > 1 <=> the wave look-ahead (kernels: `pipelined`)
        SEM_GATHER_ODD,
        SEM_OWN_FWD,
        SEM_SLOT_FWD,
    ]
    # The reader / writer kernels start their TensorAccessorArgs at index COMMON_CT_ARGS (kernels: `TensorAccessorArgs<28>`).
    assert len(common_ct) == COMMON_CT_ARGS, "common CT arg layout changed: update the kernels' accessor offset"
    reader_ct = common_ct + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct = common_ct + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct += list(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    compute_ct = list(common_ct)
    port_ct = [G, plan.num_waves, payload, MAX_HEADERS_IN_FLIGHT, SEM_OWN_STATS, SEM_STATS_LANDED, SEM_OWN_COUNTED]

    inv_w_bits = _fp32_bits(1.0 / W_global)
    eps_bits = _fp32_bits(epsilon)
    in_addr = int(input_tensor.buffer_address())
    out_addr = int(output_tensor.buffer_address())
    gamma_addr = int(gamma.buffer_address()) if has_gamma else 0
    kcfg = _kernel_config(compute_kernel_config)

    chips = _exchange_plan(mesh, build_groups(tuple(mesh.shape), cluster_axis), plan.ring)

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        peer_of = {FWD: ch["next"], BWD: ch["prev"]}
        n_out = {d: len(ch["sends"][d]) for d in (FWD, BWD)}
        n_in = {d: len(ch["entries"][d]) for d in (FWD, BWD)}
        assert max(n_in.values()) <= len(arrival_addrs), "more entries on one port than arrival counters"
        # where each peer slot lands on this chip (per link: the same direction for every link)
        slot_dir = {}
        for d in (FWD, BWD):
            for q in ch["entries"][d]:
                assert q != p and q not in slot_dir, "each peer slot must arrive exactly once"
                slot_dir[q] = d
        assert sorted(slot_dir) == [q for q in range(G) if q != p], f"chip {coord}: slots {sorted(slot_dir)}"
        pub_dirs = [d for d in (FWD, BWD) if n_out[d] > 0]
        notify_dirs = [d for d in (FWD, BWD) if n_in[d] > 0]

        program = ttnn.ProgramDescriptor()
        program.cbs = cbs
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=s, core_ranges=all_crs, initial_value=0)
            for s in (
                SEM_GATHER,
                SEM_INV,
                SEM_STATS_LANDED,
                SEM_OWN_STATS,
                SEM_OWN_COUNTED,
                SEM_GATHER_ODD,
                SEM_STATS_LANDED_BWD,
                SEM_OWN_FWD,
                SEM_SLOT_FWD,
            )
        ]

        # ---------- compute cores ----------
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for g in plan.groups:
            group_cores = [_pack(virt(c)) for c in g.cores] if r3 else []  # R3: every member's coords, index order
            for j, (core, (w0, V)) in enumerate(zip(g.cores, g.slices)):
                is_root = j < g.owners  # a root (R3-RS: an owner) — publishes / finalizes its rows every block
                links_ports = {d: virt(plan.ports[(d, g.links[j] if is_root else 0)]) for d in (FWD, BWD)}
                head = [
                    in_addr,
                    g.start_row,
                    g.rows,
                    w0,
                    V,
                    int(is_root),
                    j,
                    p,
                    landing_addr,
                    plan.seg_stride,
                    len(pub_dirs),
                    len(notify_dirs),
                ]
                pub = [_pack(links_ports[d]) for d in pub_dirs] + [SEM_LANDED[d] for d in notify_dirs]
                for d in notify_dirs:  # per notifying port: its entries' slots, in entry order
                    pub += [n_in[d]] + list(ch["entries"][d])
                slots = [0 if q == p else _pack(links_ports[slot_dir[q]]) for q in range(G)]
                offsets = list(g.offsets[j]) if is_root else [0] * plan.num_waves
                # FWD_INV root: its members' multicast rectangles (members / R1-X / R3-RS: none)
                rects = _fwd_rect_args(mesh, g.cores[1:], _NOCS[DM_NOCS[0]]) if (fwd_inv and j == 0) else [0]
                reader_rt[core.x][core.y] = head + pub + slots + offsets + group_cores + rects
                writer_rt[core.x][core.y] = [out_addr, gamma_addr, g.start_row, g.rows, w0, V]
                compute_rt[core.x][core.y] = [g.rows, inv_w_bits, eps_bits, int(is_root), V, j]

        # ---------- port cores ----------
        port_rt = ttnn.RuntimeArgs()
        for l in range(L):
            roots = [virt(g.cores[k]) for g, k in _roots(plan.groups, plan.owners) if g.links[k] == l]
            for d in (FWD, BWD):
                core = plan.ports[(d, l)]
                peer = peer_of[d]
                sends, ready_out = ch["sends"][d], ch["ready_out"][d]
                has_conn = peer is not None and (len(sends) > 0 or len(ready_out) > 0)
                pc = virt(core)  # every destination port is this same logical core on its chip
                # one pre-routed packet header per distinct destination chip (sends and readies)
                dests = sorted({(dst, hops) for _, dst, _, hops, _ in sends} | {(t, h) for t, _, h in ready_out})
                assert len(dests) <= MAX_HEADERS_IN_FLIGHT, "more destination chips than packet headers"
                args = [
                    landing_addr,
                    plan.seg_stride,
                    ready_addr,
                    len(dests),
                    n_out[d],
                    n_in[d],
                    len(roots) if n_out[d] > 0 else 0,
                    len(ready_out),
                    len({it[1] for it in sends}),  # readies to wait for: one per destination chip
                    pc.x,
                    pc.y,
                    len(roots) if n_in[d] > 0 else 0,
                    int(has_conn),
                    len(roots) if (n_out[d] > 0 and plan.num_waves > 1) else 0,  # roots to signal counted(w)
                    1 if NOTIFY_PER_ENTRY else max(1, n_in[d]),  # notify quantum (entries)
                    SEM_LANDED[d],  # this port's own landed semaphore on its roots
                ]
                args += arrival_addrs[: n_in[d]]
                assert not sends or sends[0][4] == 0, "send item 0 must be the own segment (counted(w) signal)"
                for dst, hops in dests:
                    dn = _node(mesh, dst)
                    args += [int(dn.chip_id), int(dn.mesh_id), hops]
                for q, dst, idx, hops, wait in sends:
                    args += [q, dests.index((dst, hops)), arrival_addrs[idx], wait]
                for tgt, td, hops in ready_out:
                    rc = virt(plan.ports[(td, l)])
                    args += [dests.index((tgt, hops)), rc.x, rc.y]
                args += [plan.seg_bytes[(w, l)] for w in range(plan.num_waves)]
                args += [_pack(r) for r in roots] if (n_in[d] > 0 or n_out[d] > 0) else []
                if has_conn:
                    pn = _node(mesh, peer)
                    links = ttnn.get_forwarding_link_indices(_node(mesh, coord), pn)
                    args += list(ttnn.setup_fabric_connection(_node(mesh, coord), pn, links[l], program, core))
                port_rt[core.x][core.y] = args

        reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
            core_ranges=compute_crs,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=_NOCS[DM_NOCS[0]]
            ),
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
            core_ranges=compute_crs,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=_NOCS[DM_NOCS[1]]
            ),
        )
        # DRAM NoC balance: the swapped cores' reader runs on the writer's NoC and their writer on the reader's.
        program.kernels = [
            *_split_noc_swapped(
                reader, compute_cores, swapped, ttnn.DataMovementProcessor.RISCV_1, _NOCS[DM_NOCS[0]], _NOCS[DM_NOCS[1]]
            ),
            *_split_noc_swapped(
                writer, compute_cores, swapped, ttnn.DataMovementProcessor.RISCV_0, _NOCS[DM_NOCS[1]], _NOCS[DM_NOCS[0]]
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                core_ranges=compute_crs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=kcfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port.cpp"),
                core_ranges=port_crs,
                compile_time_args=port_ct,
                runtime_args=port_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                ),
            ),
        ]
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
