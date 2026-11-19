# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm — host plan + one MeshProgramDescriptor (one ttnn.generic_op dispatch per call).

op_design.md is the specification. In short, per chip of a cluster_axis group of G chips:

* Ports (TRANSPORT): one port core per (direction, link). Its reader (NCRISC, NoC0) reads chunks of the chip's own
  slice and of the slices it relays from the MEMBERS' L1 landing slots into cb_port_chunks; its sender (BRISC, NoC1)
  ships them over the fabric into the SAME (core, offset) landing slots of the next chip's members (fabric_all_gather
  example's port pair, balanced ring / line schedule, ready fence, one fused write + increment per item).
* Members (local work): the gathered rows are dealt round-robin to row groups of k cores whose W ranges are nested in
  the slices (select_split). A member's landing slots live in cb_x_landing (CB 0, declared first on workers + ports,
  identical on every chip, so a sender can compute its peer's addresses). As rows land the member computes sum x^2,
  combines the k partials across its group (symmetric all-gather, ported from the base op), and normalizes.
* Every chip receives every raw slice and normalizes the whole gathered row itself: no partial crosses the fabric.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

# ---- CB indices (kernels/all_gather_rms_norm_common.hpp) ----
CB_X_LANDING = 0  # resident landing slots; FIRST CB, on workers + ports (same base address everywhere)
CB_GAMMA = 1  # gamma slice, compact faces-0/1 row-0 pages
CB_ONES = 2  # 1.0 reduce scaler
CB_SUMSQ = 3  # per tile-row sum of x^2
CB_PARTIAL = 4  # k > 1: per tile-row partial
CB_GATHERED = 5  # k > 1: the k partials per tile-row (owner: row-form gather tiles of the owned rows)
CB_INV_OWNED = 6  # owner combine: inv_rms of the rows this member owns
CB_INV_RMS = 7  # per tile-row rsqrt(mean + eps)
CB_X_ALIGNED = 8  # regime N: composed dense x tiles
CB_OUT = 9  # normalized output tiles
CB_PORT_CHUNKS = 10  # ports: chunk staging
CB_X_IN = 11  # native HEIGHT shard: the resident input shard (zero-copy; own-slice columns read in place)
CB_X_STAGE = 12  # regime N, bfloat8_b (Refinement 2): bf16 staging of the landed row (compute -> reader compose)
CB_X_GAMMA = 13  # prescale (Refinement 4): x * gamma of one block (T_acc), compute -> compute

# ---- Blocking-model knobs (op_design.md -> Blocking Model). Every extent below derives from these. ----
BLOCK_ROW_TILES = int(
    os.environ.get("AGRMS_BLOCK_ROW_TILES", "1")
)  # (env: perf probes) tile-rows per block (rows land progressively: one row per block)
OUT_UNIT_CAP = int(os.environ.get("AGRMS_OUT_UNIT_CAP", "16"))  # (env: perf probes) writer / compose unit (tiles)
OUT_DEPTH = 2  # cb_out_tiles depth in units
COMPOSE_DEPTH = 2  # cb_x_aligned depth in units (regime N)
STAGE_ROWS = 1  # cb_x_stage depth in landed rows (regime N bfp8; >1 needs compute look-ahead, not built)
ITEM_MIN_CHUNKS = int(
    os.environ.get("AGRMS_ITEM_MIN_CHUNKS", "4")
)  # (env: perf probes) chunks per transport item (one fused write + increment per item)
PORT_CB_BYTES = (
    int(os.environ.get("AGRMS_PORT_CB_KIB", "112")) * 1024
)  # (env: perf probes) cb_port_chunks bytes (2 batches)
PORT_GROUP_CAP = 8  # chunks per port batch
# TRANSPORT (Refinement 3): the far slice of a balanced ring is split between the two directions by tile-row parity
# (part 1 = even rows, part 2 = odd rows, each over the link's whole column range) instead of by columns (each half of
# the link's columns), so its packets carry the link's full run (6 tiles instead of 3 on the focus case). 0 = the
# Phase 0 column split (A/B only).
FAR_ROW_SPLIT = os.environ.get("AGRMS_FAR_ROW_SPLIT", "1") == "1"
RELAY_LAG = int(
    os.environ.get("AGRMS_RELAY_LAG", "2")
)  # (env: perf probes) steps between a relay item's upstream send and its relay (>= 1)
# OVERLAP (Refinement 4): x * gamma is computed while the k partials are exchanged (the combine wait), so only one
# multiply (x*gamma) * inv_rms is left after the combine. Regime A, k > 1, gamma, round-robin split (prescale_on).
# 0 = the fused two-multiply normalize after the combine (A/B only).
PRESCALE_GAMMA = os.environ.get("AGRMS_PRESCALE", "1") == "1"
K_SYM_MAX = int(
    os.environ.get("AGRMS_K_SYM_MAX", "8")
)  # (env: perf probes) symmetric all-gather of partials up to this k; per-row owner combine beyond (base op)
ROW_GATHER = 32  # owner combine: members per row-form gather tile (one 32-datum row each)
# Local work (Refinement 5): the symmetric combine ships each member's partial in ROW form (the owner combine's
# ones-row x sumsq^T collapse: row 0 = the 32 row sums) as two face rows (2 x 64 B at fp32) into row m of every peer's
# gather tile, instead of the column-0 faces 0/2 (2 x 1 KiB). Same k^2 messages per tile-row, 1/16 of the bytes: the
# exchange was NoC-bound (every member core sends at once). 0 = the column-form exchange (A/B only).
ROW_PARTIALS = os.environ.get("AGRMS_ROW_PARTIALS", "1") == "1"
OCC_TILES_PER_MEMBER = 4  # member width floor (except the smallest candidate)
COMBINE_TILE_COST = 3  # per tile-row combine cost (x-tile equivalents), base op
OWNER_MEMBER_COST = 0.4  # owner combine: per member (x-tile equivalents), base op
TILIZED_WTILE_NS = 310  # per W tile of a tile-row, base op
KEEP_UP_FRACTION = 0.5
# Native sharded placement (Refinement 1) is taken when its shard-fixed split keeps up with the transport: per-core
# work (same cost model) <= NATIVE_KEEP_UP_FRACTION * T_xfer. Measured (QB2 1x4 ring L2, fastest chip): HEIGHT
# 640x384 keeps up (0.83) native 34.9 vs fallback 41.4 us; HEIGHT 2048x384 (0.39) 68.2 vs 93.4; HEIGHT 128x512 (1.8)
# 27.6 vs 20.2; WIDTH 64x512 k=64 (1.9) 32.6 vs 14.3; WIDTH 32x4096 k=103 (1.1) 32.7 vs 24.9.
NATIVE_KEEP_UP_FRACTION = 1.0
# Share of the transport byte term that scales with the packet count (the ports are packet-rate bound, Refinement 3).
# Calibrated (Refinement 4) on 640x1792 ring L2: k = 12 cuts the link runs into 10 packets per slice-row instead of 8
# (ratio 1.25) and its TRANSPORT-only ablation measured 76.2 vs 70.0 us at k = 8: 6.2 / (0.25 * 44.6 us) = 0.55.
PACKET_TIME_SHARE = 0.55
XFER_FIT = {True: (8.4e-6, 77.2e9), False: (9.8e-6, 42.1e9)}  # (fixed s, bytes/s per link): fabric_all_gather fits
PARTIAL_FACES_ONLY = True
GAMMA_READ_ALIGN = 64  # BH DRAM read alignment: cb_gamma page stride (host) and gamma reads (writer CT arg)
MAX_RUNTIME_ARGS = 341
# Perf probes only (never set by the op): AGRMS_ABLATE=EMPTY | LOCAL | TRANSPORT stubs one side of the program
# (kernels/*: AGRMS_ABLATE_*). EMPTY: every kernel returns at once (launch floor). LOCAL: no transport (ports idle,
# members do not wait for landings). TRANSPORT: members only load, signal and wait for their landings. NOREAD: the port
# reader skips its NoC reads (pushes kept; output garbage) to size the port read stage.
PERF_ABLATE = os.environ.get("AGRMS_ABLATE", "")
# Perf probes only: AGRMS_PROFILE_ZONES=1 compiles the per-stage device zones in (kernels: AGRMS_ZONE).
PROFILE_ZONES = os.environ.get("AGRMS_PROFILE_ZONES", "") == "1"
# Test hook only (debug tests pin a split): AGRMS_FORCE_K = k restricts select_split to that candidate.
FORCE_K = int(os.environ.get("AGRMS_FORCE_K", "0"))
# Perf A/B only: AGRMS_NATIVE_SHARD=0 sends sharded tensors down the TensorAccessor fallback (the Phase 0 path).
NATIVE_SHARD = os.environ.get("AGRMS_NATIVE_SHARD", "1") == "1"
# presend_critical_path experiment knobs (port kernels; perf_presend.py).
PRESEND_A = os.environ.get("AGRMS_PRESEND_A", "0") == "1"  # route one header, copy the rest
PRESEND_DRAM_ITEMS = int(os.environ.get("AGRMS_PRESEND_DRAM_ITEMS", "0"))  # own items read from DRAM by the port
PRESEND_SPLIT_FENCE = os.environ.get("AGRMS_PRESEND_SPLIT_FENCE", "0") == "1"  # ready out before everything else
PRESEND_HOSTCOUNT = os.environ.get("AGRMS_PRESEND_HOSTCOUNT", "0") == "1"  # sender's chunk count from the host
PRESEND_LAZY = os.environ.get("AGRMS_PRESEND_LAZY", "0") == "1"  # headers 1.. copied at their first use
PRESEND_DMA_COPY = os.environ.get("AGRMS_PRESEND_DMA_COPY", "0") == "1"  # header copies by local NoC writes
PRESEND_DRAM_PUSH1 = os.environ.get("AGRMS_PRESEND_DRAM_PUSH1", "0") == "1"  # DRAM items: push every chunk
# lever (b1): per-item readiness of the first own items: the members of the groups of own item j < EARLY_ITEMS
# signal semaphore SEM_EARLY + j after their local row 0 (round-robin rows only)
PRESEND_EARLY_ITEMS = int(os.environ.get("AGRMS_PRESEND_EARLY_ITEMS", "0"))
PRESEND_EARLY_OPEN = os.environ.get("AGRMS_PRESEND_EARLY_OPEN", "0") == "1"  # open_start before routing header 0


def _div_up(a, b):
    return (a + b - 1) // b


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def acc_dtype(fp32_dest_acc_en):
    """Intermediate (DEST-written) CB format follows the DEST width (base op Refinement 1)."""
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def datum_bytes(dtype):
    return 1 if dtype == ttnn.bfloat8_b else ttnn.tile_size(dtype) // (TILE * TILE)


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Geometry:
    G: int
    W_local: int
    M: int
    num_images: int
    R_t: int  # tile-rows per chip (per image ceil)
    Wt_lp: int  # tiles per slice
    land_w_tiles: int  # G * Wt_lp
    out_w_tiles: int  # ceil(G * W_local / 32)
    W: int  # logical gathered width
    regime_n: bool


def derive_geometry(shape, G) -> Geometry:
    shape = [int(d) for d in shape]
    W_local, M = shape[-1], shape[-2]
    num_images = math.prod(shape[:-2]) if len(shape) > 2 else 1
    Wt_lp = _div_up(W_local, TILE)
    return Geometry(
        G=G,
        W_local=W_local,
        M=M,
        num_images=num_images,
        R_t=num_images * _div_up(M, TILE),
        Wt_lp=Wt_lp,
        land_w_tiles=G * Wt_lp,
        out_w_tiles=_div_up(G * W_local, TILE),
        W=G * W_local,
        regime_n=(W_local % TILE) != 0,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Groups, schedules, links (adapted from the fabric_all_gather example's plan())
# ---------------------------------------------------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def schedule(p, G, ring, balance=True):
    """(entries toward p+1, entries toward p-1), own slice first: (slice position, part). part 0 = whole, 1 / 2 = the
    first / second half of the link's columns (the far slice of an even ring split between the directions)."""
    if ring and balance and G % 2 == 0 and G >= 4:
        h = G // 2
        fwd = [((p - i) % G, 1 if i == h - 1 else 0) for i in range(h)]
        bwd = [((p + i) % G, 2 if i == h - 1 else 0) for i in range(h)]
        return fwd, bwd
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [((p - i) % G, 0) for i in range(kf)], ([((p + i) % G, 0) for i in range(kb)] if kb > 0 else [])
    fwd = [(p - i, 0) for i in range(p + 1)] if p < G - 1 else []
    bwd = [(p + i, 0) for i in range(G - p)] if p > 0 else []
    return fwd, bwd


DIRS = ("fwd", "bwd")
_OPP = {"fwd": "bwd", "bwd": "fwd"}


def link_cols(Wt_lp, L, l, part, row_split=False):
    """(c0, c1, rf): columns [c0, c1) of a slice that link l carries for an entry part, and the row filter rf (0 = every
    tile-row, 1 / 2 = even / odd rows). Part halves split the link's range by columns, or (row_split) by rows."""
    a, b = (l * Wt_lp) // L, ((l + 1) * Wt_lp) // L
    if part == 0:
        return a, b, 0
    if row_split:
        return a, b, part
    h = a + (b - a) // 2
    return (a, h, 0) if part == 1 else (h, b, 0)


def entry_has_row(rf, r):
    return rf == 0 or (r & 1) == rf - 1


def discovered_links(mesh_device, cluster_axis, ring):
    mesh_shape = tuple(mesh_device.shape)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    n = None
    for grp in build_groups(mesh_shape, cluster_axis):
        G = len(grp)
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring:
            hops.append((grp[-1], grp[0]))
        for a, b in hops:
            for s, d in ((a, b), (b, a)):
                c = len(ttnn.get_forwarding_link_indices(node(s), node(d)))
                n = c if n is None else min(n, c)
    return n or 0


# ---------------------------------------------------------------------------------------------------------------------
# Port placement (host-side; no probe program). Adapted from the fabric_all_gather example.
# ---------------------------------------------------------------------------------------------------------------------

_ETH_MAP_CACHE = {}


def _arch_yaml(mesh_device):
    import yaml

    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    name = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh_device.arch()).split(".")[-1]
    ]
    return yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", name)))


def _eth_physical_lists(mesh_device):
    """{asic unique id: ([physical (x, y) of Ethernet core i], translated, translated->physical tensix x)}."""
    if id(mesh_device) in _ETH_MAP_CACHE:
        return _ETH_MAP_CACHE[id(mesh_device)][1]
    import yaml

    desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
    soc = _arch_yaml(mesh_device)
    arch_eth = [tuple(int(v) for v in e.split("-")) for e in soc["eth"]]
    tensix_cols = sorted({int(w.split("-")[0]) for w in soc["functional_workers"]})
    lists = {}
    for chip, uid in desc.get("chip_unique_ids", {}).items():
        h = desc.get("harvesting", {}).get(chip, {})
        mask = int(h.get("eth_harvesting_mask", 0))
        translated = bool(h.get("noc_translation", False))
        tmask = int(h.get("harvest_mask", 0))
        live = [x for i, x in enumerate(tensix_cols) if not (tmask >> i) & 1]
        t2p = {tensix_cols[k]: px for k, px in enumerate(live)} if translated else {}
        lists[int(uid)] = ([e for i, e in enumerate(arch_eth) if not (mask >> i) & 1], translated, t2p)
    grid = soc["grid"]
    _ETH_MAP_CACHE[id(mesh_device)] = (mesh_device, (lists, (int(grid["x_size"]), int(grid["y_size"]))))
    return _ETH_MAP_CACHE[id(mesh_device)][1]


def _chip_maps(mesh_device, coord):
    lists, grid = _eth_physical_lists(mesh_device)
    node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(node.mesh_id), int(node.chip_id)))
    return lists.get(uid, (None, False, {})), grid


def _worker_nearest_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Free logical core with the fewest NoC1 hops (-Y then -X, wrapping) to the Ethernet core at physical eth_xy."""
    (_, _, t2p), (gx, gy) = _chip_maps(mesh_device, coord)
    ex, ey = eth_xy
    best = None
    for x, y in sorted(allowed):
        if (x, y) in taken:
            continue
        v = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        px = t2p.get(v.x, v.x)
        score = ((v.y - ey) % gy + (px - ex) % gx, v.y, px)
        if best is None or score < best[0]:
            best = (score, (x, y))
    return best[1] if best is not None else None


def _first_free(taken, allowed):
    """Simple placement: the free core of the lowest logical row (nearest the Ethernet row), left to right."""
    for x, y in sorted(allowed, key=lambda c: (c[1], c[0])):
        if (x, y) not in taken:
            return (x, y)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Plan: groups, schedules, links, ports (cached per mesh / axis / topology / links)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ChipPlan:
    coord: tuple
    p: int
    G: int
    nxt: tuple | None
    prv: tuple | None
    sends: dict  # direction -> [(slice, part)]
    links: dict = field(default_factory=dict)  # direction -> [link index] (L of them)
    ports: dict = field(default_factory=dict)  # (direction, l) -> logical (x, y)


_PLAN_CACHE = {}


def allowed_cores(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return {(x, y) for x in range(grid.x) for y in range(grid.y)}


def plan(mesh_device, cluster_axis, ring, L, avoid=frozenset()):
    """avoid: logical cores the ports must not take (native shard placement: the member shard cores)."""
    key = (id(mesh_device), cluster_axis, ring, L, avoid)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key][1]
    mesh_shape = tuple(mesh_device.shape)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    groups = build_groups(mesh_shape, cluster_axis)
    G = len(groups[0])
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            nxt = grp[(p + 1) % G] if (ring or p < G - 1) else None
            prv = grp[(p - 1) % G] if (ring or p > 0) else None
            fwd, bwd = schedule(p, G, ring)
            chips[coord] = ChipPlan(coord=coord, p=p, G=G, nxt=nxt, prv=prv, sends={"fwd": fwd, "bwd": bwd})
    for ch in chips.values():
        for d in DIRS:
            peer = ch.nxt if d == "fwd" else ch.prv
            if peer is None:
                ch.sends[d] = []
                continue
            links = ttnn.get_forwarding_link_indices(node(ch.coord), node(peer))
            if len(links) < L:
                raise ValueError(
                    f"all_gather_rms_norm: {ch.coord} -> {peer} has {len(links)} usable link(s), {L} requested"
                )
            ch.links[d] = list(links[:L])
    # placement: each port next to its link's Ethernet core (fewest NoC1 hops); receive-only ports anywhere free
    allowed = allowed_cores(mesh_device) - set(avoid)
    for ch in chips.values():
        taken = set()
        pending = []
        for d in DIRS:
            peer = ch.nxt if d == "fwd" else ch.prv
            for l in range(L):
                core = None
                if peer is not None:
                    try:
                        scratch = ttnn.ProgramDescriptor()
                        chan = int(
                            ttnn.setup_fabric_connection(
                                node(ch.coord), node(peer), ch.links[d][l], scratch, ttnn.CoreCoord(0, 0)
                            )[0]
                        )
                        (eth_list, _, _), _ = _chip_maps(mesh_device, ch.coord)
                        if eth_list is not None and chan < len(eth_list):
                            core = _worker_nearest_noc1(mesh_device, ch.coord, eth_list[chan], taken, allowed)
                    except Exception:  # noqa: BLE001 — placement is a perf choice; fall back to simple placement
                        core = None
                if core is None:
                    pending.append((d, l))
                else:
                    ch.ports[(d, l)] = core
                    taken.add(core)
        for d, l in pending:
            core = _first_free(taken, allowed)
            if core is None:
                raise ValueError("all_gather_rms_norm: not enough cores for the ports")
            ch.ports[(d, l)] = core
            taken.add(core)
    _PLAN_CACHE[key] = (mesh_device, (chips, G))
    return chips, G


# ---------------------------------------------------------------------------------------------------------------------
# Work split (select_split): the base op's split rule re-run on the grid left after the ports
# ---------------------------------------------------------------------------------------------------------------------


def member_ranges(geo: Geometry, k: int):
    """Landing-tile boundaries ws[0..k] of the k members of a group, nested in slices."""
    G, Wt = geo.G, geo.Wt_lp
    if k <= G:
        m = G // k
        return [i * m * Wt for i in range(k + 1)]
    ks = k // G
    ws = [q * Wt + (j * Wt) // ks for q in range(G) for j in range(ks)]
    return ws + [G * Wt]


def nested_candidates(geo: Geometry):
    if geo.regime_n:
        return [1]
    G, Wt = geo.G, geo.Wt_lp
    cands = sorted({G // m for m in range(1, G + 1) if G % m == 0} | {G * ks for ks in range(2, Wt + 1)})
    out = []
    for k in cands:
        ws = member_ranges(geo, k)
        wmin = min(ws[i + 1] - ws[i] for i in range(k))
        if k == cands[0] or wmin >= OCC_TILES_PER_MEMBER:
            out.append(k)
    return out


@dataclass
class Split:
    k: int
    ws: list
    w_pitch: int
    n_groups: int
    rows_max: int
    blocks_max: int
    combine_owner: bool


@dataclass
class Sizes:
    T_in: int
    T_acc: int
    T_g: int
    g_tail: int
    run_pages: int
    group: int
    has_gamma: bool
    T_stage: int = 0  # regime N bfp8: bf16 staging tile bytes (0 = no staging, compose on the input dtype)


def gamma_page_layout(gamma_dtype):
    """(cb_gamma page stride, tail pages): base op compact cb_gamma (Perf 2) — a page only needs row 0 of faces 0/1,
    so the stride is the smallest multiple of 64 B whose per-page write footprints never overlap across pages."""
    elem = datum_bytes(gamma_dtype)
    exp = ttnn.tile_size(gamma_dtype) - TILE * TILE * elem
    face = (TILE * TILE * elem) // 4
    full = exp + 2 * face
    foot = [(0, exp + GAMMA_READ_ALIGN), (exp + face, GAMMA_READ_ALIGN)]

    def disjoint(s):
        for j in range(1, full // s + 2):
            for o1, l1 in foot:
                for o2, l2 in foot:
                    a, b = o2 + j * s, o2 + j * s + l2
                    if a < o1 + l1 and o1 < b:
                        return False
        return True

    for s in range(GAMMA_READ_ALIGN, full, GAMMA_READ_ALIGN):
        if disjoint(s):
            return s, _div_up(full - s, s)
    return full, 0


def out_unit_of(geo: Geometry, width):
    return min(width, OUT_UNIT_CAP)


def row_partials_on(split: Split):
    """Symmetric combine with row-form partials (writer / compute CT `row_partials`)."""
    return ROW_PARTIALS and 1 < split.k and not split.combine_owner


def prescale_on(geo: Geometry, sz: Sizes, split: Split, native: bool = False):
    """x * gamma during the combine wait (compute CT `prescale`): needs a combine wait to hide under (k > 1), gamma, and
    the resident landing block as x (regime A, round-robin split; native shards normalize in segments)."""
    return PRESCALE_GAMMA and sz.has_gamma and not geo.regime_n and split.k > 1 and not native


def cb_plan(geo: Geometry, sz: Sizes, split: Split, native: bool = False):
    """{cb index: (pages, page bytes)} of a worker core (cb_port_chunks is the ports' only extra CB)."""
    b = BLOCK_ROW_TILES
    k = split.k
    out_w = geo.out_w_tiles if geo.regime_n else split.w_pitch
    unit = out_unit_of(geo, out_w)
    plan_ = {
        CB_X_LANDING: (split.blocks_max * b * split.w_pitch, sz.T_in),
        CB_ONES: (1, ttnn.tile_size(ttnn.bfloat16)),
        CB_SUMSQ: (b, sz.T_acc),
        CB_INV_RMS: (b, sz.T_acc),
        CB_OUT: (OUT_DEPTH * unit, sz.T_in),
    }
    if sz.has_gamma:
        plan_[CB_GAMMA] = (out_w + sz.g_tail, sz.T_g)
    if k > 1:
        plan_[CB_PARTIAL] = (b, sz.T_acc)
        if split.combine_owner:
            owned = _div_up(b, k)  # tile-rows a member owns per block
            plan_[CB_GATHERED] = (owned * _div_up(k, ROW_GATHER), sz.T_acc)  # uniform per block: lockstep slots
            plan_[CB_INV_OWNED] = (owned, sz.T_acc)
        else:
            depth_g = 2 if split.blocks_max > 1 else 1  # lockstep slot reuse (base op mechanism cap)
            per_row = _div_up(k, ROW_GATHER) if row_partials_on(split) else k  # gather tiles vs k column tiles
            plan_[CB_GATHERED] = (depth_g * per_row * b, sz.T_acc)
    if prescale_on(geo, sz, split, native):
        plan_[CB_X_GAMMA] = (b * split.w_pitch, sz.T_acc)  # one block of x * gamma
    if geo.regime_n:
        # staged (bfp8): the compose writes 2-byte datums, so cb_x_aligned holds bf16 tiles
        plan_[CB_X_ALIGNED] = (COMPOSE_DEPTH * unit, sz.T_stage or sz.T_in)
        if sz.T_stage:
            plan_[CB_X_STAGE] = (STAGE_ROWS * geo.land_w_tiles, sz.T_stage)
    return plan_


def _plan_bytes(p):
    return sum(pages * page for pages, page in p.values())


def port_footprint(sz: Sizes, split: Split):
    return split.blocks_max * BLOCK_ROW_TILES * split.w_pitch * sz.T_in + 2 * sz.group * sz.run_pages * sz.T_in


def transfer_time(geo: Geometry, sz: Sizes, ring: bool, L: int, packet_ratio: float = 1.0):
    """Modelled transport time (s): fabric_all_gather fit for the G - 1 slices a chip receives. packet_ratio scales the
    byte term for a member split that cuts the link runs into more packets (the ports are packet-rate bound)."""
    S = geo.M * geo.num_images * geo.W_local * sz.T_in / (TILE * TILE)
    t0, rate = XFER_FIT[ring]
    return t0 + (1.0 + PACKET_TIME_SHARE * (packet_ratio - 1.0)) * (geo.G - 1) * S / (rate * L)


def packets_per_row(geo: Geometry, ws, L: int, run_pages: int):
    """Fabric packets per gathered row: every link part of every slice, cut at the member boundaries ws into runs of
    <= run_pages tiles (the port walk's chunking; see cpr_of in create_mesh_program_descriptor)."""
    n = 0
    for q in range(geo.G):
        for l in range(L):
            c0, c1 = link_cols(geo.Wt_lp, L, l, 0)[:2]
            t0, t1 = q * geo.Wt_lp + c0, q * geo.Wt_lp + c1
            for m in range(len(ws) - 1):
                a, b = max(t0, ws[m]), min(t1, ws[m + 1])
                if a < b:
                    n += _div_up(b - a, run_pages)
    return n


def combine_cost(k: int, owner: bool):
    """Per tile-row combine cost of a k-member group (x-tile equivalents, base op)."""
    return 0 if k == 1 else (COMBINE_TILE_COST if not owner else COMBINE_TILE_COST + OWNER_MEMBER_COST * k)


def native_keeps_up(geo: Geometry, sz: Sizes, split: Split, ring: bool, L: int):
    total = split.rows_max * (split.w_pitch + combine_cost(split.k, split.combine_owner))
    return total * TILIZED_WTILE_NS * 1e-9 <= NATIVE_KEEP_UP_FRACTION * transfer_time(geo, sz, ring, L)


def select_split(geo: Geometry, sz: Sizes, n_workers: int, budget: int, ring: bool, L: int):
    """op_design.md -> Regime / split selection. Objective (OVERLAP, Refinement 4): the estimated end of the call =
    transport time of this split (packet-aware: member boundaries that cut the link runs into more packets slow the
    packet-rate-bound ports) + the tail after the last landing, among the splits that keep up with that transport;
    else the smallest total; else RuntimeError. With equal packet counts this is the Phase 0 tail objective."""
    p_ref = packets_per_row(geo, [0, geo.land_w_tiles], L, sz.run_pages)  # unsplit link runs
    k_cap = min((MAX_RUNTIME_ARGS - 16) // 2, n_workers)  # port reader RT args carry 2k + O(G) values
    fits, keeps = [], []
    for k in nested_candidates(geo):
        if k > k_cap or (FORCE_K and k != FORCE_K):
            continue
        ws = member_ranges(geo, k)
        w_pitch = max(ws[i + 1] - ws[i] for i in range(k))
        n_groups = min(n_workers // k, geo.R_t)
        if n_groups == 0:
            continue
        rows_max = _div_up(geo.R_t, n_groups)
        blocks_max = _div_up(rows_max, BLOCK_ROW_TILES)
        owner = k > K_SYM_MAX
        split = Split(k, ws, w_pitch, n_groups, rows_max, blocks_max, owner)
        if _plan_bytes(cb_plan(geo, sz, split)) > budget or port_footprint(sz, split) > budget:
            continue
        C = combine_cost(k, owner)
        total = rows_max * (w_pitch + C)
        tail = BLOCK_ROW_TILES * (w_pitch + C)
        T_xfer = transfer_time(geo, sz, ring, L, packets_per_row(geo, ws, L, sz.run_pages) / p_ref)
        fits.append((total, rows_max, split))
        if total * TILIZED_WTILE_NS * 1e-9 <= KEEP_UP_FRACTION * T_xfer:
            keeps.append((T_xfer + tail * TILIZED_WTILE_NS * 1e-9, rows_max, split))
    if keeps:
        return min(keeps, key=lambda t: (t[0], t[1]))[2]
    if fits:
        return min(fits, key=lambda t: (t[0], t[1]))[2]
    raise RuntimeError(
        f"all_gather_rms_norm: no admissible split fits L1 (budget {budget} B) for per-device shape "
        f"rows {geo.R_t} x slice {geo.Wt_lp} tiles x G {geo.G} (dram_landing regime is deferred)"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Native sharded placement (Refinement 1): the shard grid IS the work split (base op Refinements 2 / 4)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class NativeShard:
    kind: str  # "height" | "width"
    split: Split
    cores: list  # member cores in member order (worker index g * k + m), = the output shard cores in shard order
    rows_per_group: int  # HEIGHT: tile-rows per shard (group gi holds rows [gi * rpg, ...)); WIDTH: 0 (one group)
    group_rows: list  # tile-rows of each group
    out_pitch: int  # tiles per row of the output shard (WIDTH: a ragged last member pushes pad pages up to it)


def _ordered_shard_cores(grid, row_major: bool):
    """Cores of a shard grid in shard-assignment order (row-major or column-major) — base op."""
    cores = set()
    for r in grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.add((x, y))
    return sorted(cores, key=(lambda c: (c[1], c[0])) if row_major else (lambda c: (c[0], c[1])))


def _shard_geom(tensor):
    mc = tensor.memory_config()
    ss = mc.shard_spec
    if not mc.is_sharded() or ss is None or mc.buffer_type != ttnn.BufferType.L1:
        return None
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    if sh % TILE or sw % TILE:
        return None
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    return mc.memory_layout, sh // TILE, sw // TILE, _ordered_shard_cores(ss.grid, row_major)


def native_shard_split(geo: Geometry, input_tensor, output_tensor):
    """The native (shard-resident) split, or None when the shard geometry needs the TensorAccessor fallback.

    Predicate (op_design.md -> Regimes, sharded_zero_copy):
      HEIGHT: input and output HEIGHT_SHARDED with the same row partition (shard height, grid, orientation) and
        full-width shards. One group per shard (k = 1, contiguous rows), member = the shard core. The own-slice
        columns are read in place from the input shard (cb_x_in); the output is packed into the output shard.
      WIDTH: output WIDTH_SHARDED with full-height shards, W_local % 32 == 0 (regime A): one group, k = output
        shard columns, member m = output shard core m holding gathered tiles [m * pw, (m + 1) * pw). The own slice
        is gathered from the input shards (cross-core reads); the output is packed into the output shard.
      Both: the ports must be placeable off the member cores (checked by the caller) and the CBs must fit L1.
    """
    gi, go = _shard_geom(input_tensor), _shard_geom(output_tensor)
    if gi is None or go is None or gi[0] != go[0]:
        return None
    ml, oh, ow, ocores = go
    if ml == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        _, ih, iw, icores = gi
        if ih != oh or icores != ocores or iw != geo.Wt_lp or ow != geo.out_w_tiles:
            return None
        n = _div_up(geo.R_t, oh)
        if n > len(ocores):
            return None
        land_w = geo.land_w_tiles
        rows = [min(oh, geo.R_t - g * oh) for g in range(n)]
        split = Split(1, [0, land_w], land_w, n, oh, _div_up(oh, BLOCK_ROW_TILES), False)
        return NativeShard("height", split, ocores[:n], oh, rows, geo.out_w_tiles)
    if ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if geo.regime_n or oh != geo.R_t:
            return None
        total = geo.land_w_tiles
        k = _div_up(total, ow)
        if k > len(ocores) or k > (MAX_RUNTIME_ARGS - 16) // 2:
            return None
        ws = [min(m * ow, total) for m in range(k + 1)]
        split = Split(k, ws, ow, 1, geo.R_t, _div_up(geo.R_t, BLOCK_ROW_TILES), k > K_SYM_MAX)
        return NativeShard("width", split, ocores[:k], 0, [geo.R_t], ow)
    return None


def cb_l1_budget(device) -> int:
    """Free L1 per core for CBs: the allocator base up to the lowest allocated L1 buffer (base op)."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else int(view.total_bytes_per_bank)


# ---------------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------------


def _rects(cores):
    """Merge logical (x, y) cores into rectangles (one CoreRange per core makes the program dispatch-bound)."""
    cores = set(cores)
    runs = {}
    for y in sorted({y for _, y in cores}):
        xs = sorted(x for x, yy in cores if yy == y)
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            runs.setdefault((start, prev), []).append(y)
            if x is not None:
                start = prev = x
    ranges = []
    for (x0, x1), ys in runs.items():
        y0 = prev_y = ys[0]
        for y in ys[1:] + [None]:
            if y is not None and y == prev_y + 1:
                prev_y = y
                continue
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, prev_y)))
            if y is not None:
                y0 = prev_y = y
    return ttnn.CoreRangeSet(ranges)


def _cb(index, pages, page_bytes, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _shard_backed_cb(index, tensor, core_ranges):
    """CB aliasing a TILE tensor's resident shard (zero-copy: the shard IS the CB's L1) — base op."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    cb.core_ranges = core_ranges
    return cb


def _pack_entry(q, c0, c1, rf):
    """kernels/all_gather_rms_norm_port_walk.hpp unpack_entry: rf[31:30] q[29:24] c0[23:12] c1[11:0]."""
    assert q < 64 and c0 < 4096 and c1 < 4096 and rf < 4
    return (rf << 30) | (q << 24) | (c0 << 12) | c1


def _segment_args(entries, ws, Wt_lp):
    """Per send entry (kernels/all_gather_rms_norm_port_walk.hpp Entry): the members its columns overlap, one word
    each: member m [31:24], landing-tile offset inside the member's row [23:12], run length [11:0]. Returns the
    port-kernel arg block: (first << 16 | count) per entry, the segment count, the segments."""
    idx, segs = [], []
    for q, c0, c1, _ in entries:
        t0, t1 = q * Wt_lp + c0, q * Wt_lp + c1
        first = len(segs)
        for m in range(len(ws) - 1):
            a, b = max(t0, ws[m]), min(t1, ws[m + 1])
            if a < b:
                assert m < 256 and a - ws[m] < 4096 and b - a < 4096
                segs.append((m << 24) | ((a - ws[m]) << 12) | (b - a))
        assert first < 65536 and len(segs) - first < 65536
        idx.append((first << 16) | (len(segs) - first))
    return idx + [len(segs)] + segs


_READY_SEM_CACHE = {}


def ready_semaphore(mesh_device, port_cores):
    """One cached global semaphore (initial 0) on the union of port cores; synchronized once at creation so that it
    holds 0 on every chip before any chip's first program (never inside a trace)."""
    key = (id(mesh_device), tuple(sorted(port_cores)))
    if key not in _READY_SEM_CACHE:
        crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(port_cores)]
        )
        sem = ttnn.create_global_semaphore(mesh_device, crs, 0)
        ttnn.synchronize_device(mesh_device)
        _READY_SEM_CACHE[key] = (mesh_device, sem)
    return _READY_SEM_CACHE[key][1]


def create_mesh_program_descriptor(
    input_tensor, gamma, output_tensor, *, cluster_axis, ring, L, epsilon, compute_kernel_config
):
    mesh_device = input_tensor.device()
    G = tuple(mesh_device.shape)[cluster_axis]
    geo = derive_geometry(input_tensor.shape, G)
    has_gamma = gamma is not None
    x_dtype = input_tensor.dtype
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    t_acc = acc_dtype(fp32_dest)
    T_in = ttnn.tile_size(x_dtype)
    run_pages = max(1, int(ttnn.get_tt_fabric_max_payload_size_bytes()) // T_in)
    group = max(1, min(PORT_GROUP_CAP, PORT_CB_BYTES // (2 * run_pages * T_in)))
    T_g, g_tail = gamma_page_layout(gamma.dtype) if has_gamma else (0, 0)
    # regime N on bfp8 (Refinement 2): datums cannot be shifted inside shared-exponent groups, so compute unpacks the
    # landed row into a bf16 staging row and the reader composes from it (2-byte datums)
    stage = geo.regime_n and x_dtype == ttnn.bfloat8_b
    T_stage = ttnn.tile_size(ttnn.bfloat16) if stage else 0
    sz = Sizes(T_in, ttnn.tile_size(t_acc), T_g, g_tail, run_pages, group, has_gamma, T_stage)

    # ---- native sharded placement (Refinement 1): the shard grid fixes the split; the ports avoid its cores ----
    allowed = allowed_cores(mesh_device)
    budget = cb_l1_budget(mesh_device)
    native = native_shard_split(geo, input_tensor, output_tensor) if NATIVE_SHARD else None
    if native is not None:
        resident = {CB_OUT} | ({CB_X_IN} if native.kind == "height" else set())  # shard-backed: not CB budget
        nplan = cb_plan(geo, sz, native.split, native=True)
        fits = sum(pg * pb for i, (pg, pb) in nplan.items() if i not in resident) <= budget
        if not fits or port_footprint(sz, native.split) > budget or not set(native.cores) <= allowed:
            native = None
        elif not native_keeps_up(geo, sz, native.split, ring, L):
            native = None  # the shard-fixed split would bind the call: the round-robin split is faster
    chips = None
    if native is not None:
        try:
            chips, _ = plan(mesh_device, cluster_axis, ring, L, frozenset(native.cores))
        except ValueError:  # not enough cores off the shard grid for the ports: TensorAccessor fallback
            native = None
    if chips is None:
        chips, _ = plan(mesh_device, cluster_axis, ring, L)
    native_h = native is not None and native.kind == "height"

    # ---- worker set: allowed cores minus the union of every chip's port cores (identical on every chip) ----
    port_union = {c for ch in chips.values() for c in ch.ports.values()}
    workers = sorted(allowed - port_union, key=lambda c: (c[1], c[0]))
    if not workers:
        raise ValueError("all_gather_rms_norm: no worker core left after the ports")
    if native is not None:
        split = native.split
        active = list(native.cores)  # worker index g * k + m = shard order
        assert not (set(active) & port_union)
        rows_per_group = native.rows_per_group
        group_rows = native.group_rows
    else:
        split = select_split(geo, sz, len(workers), budget, ring, L)
        active = workers[: split.n_groups * split.k]  # worker index g * k + m
        rows_per_group = 0  # round-robin rows: row r -> group r % n_groups
        group_rows = [len(range(gi, geo.R_t, split.n_groups)) for gi in range(split.n_groups)]
    k, ws, w_pitch, n_groups = split.k, split.ws, split.w_pitch, split.n_groups
    virt = lambda c: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
    noc_xy = [(lambda v: v.x | (v.y << 16))(virt(c)) for c in active]

    # transport item rows: >= ITEM_MIN_CHUNKS chunks per item (one fused increment per item)
    def cpr_of(q, c0, c1):
        t0, t1 = q * geo.Wt_lp + c0, q * geo.Wt_lp + c1
        n = 0
        for m in range(k):
            a, b = max(t0, ws[m]), min(t1, ws[m + 1])
            if a < b:
                n += _div_up(b - a, run_pages)
        return n

    cpr = max(cpr_of(0, *link_cols(geo.Wt_lp, L, l, 0)[:2]) for l in range(L))
    xfer_rows = max(1, _div_up(ITEM_MIN_CHUNKS, max(cpr, 1)))

    out_w = geo.out_w_tiles if geo.regime_n else w_pitch  # tiles normalized / stored per row (pitch)
    unit = out_unit_of(geo, out_w)
    plan_ = cb_plan(geo, sz, split, native=native is not None)
    prescale = prescale_on(geo, sz, split, native=native is not None)
    if native is not None:
        plan_.pop(CB_OUT)  # backed on the output shard below

    # ---- semaphores (program, on every core of every chip's union set; ids identical everywhere) ----
    n_streams = 2 * L
    SEM_PARTIALS = n_streams
    SEM_LOCAL_LOADED = n_streams + 1
    SEM_ARRIVAL = n_streams + 2
    SEM_INV = n_streams + 3  # owner combine: inv_rms returns
    SEM_PARTIALS_ODD = n_streams + 4  # partials of odd blocks (writer: one counter per block parity)
    NUM_SEMS = n_streams + 5
    SEM_EARLY = NUM_SEMS
    # early items: item j's rows [j * xfer_rows, (j + 1) * xfer_rows) all on local row 0 (round-robin, row r -> group r)
    n_early = 0
    if PRESEND_EARLY_ITEMS > 0 and native is None:
        n_early = min(PRESEND_EARLY_ITEMS, split.n_groups // xfer_rows, _div_up(geo.R_t, xfer_rows))
        n_early = min(n_early, 16 - NUM_SEMS)
    NUM_SEMS += n_early

    all_cores = sorted(set(workers) | port_union, key=lambda c: (c[1], c[0]))
    all_crs = _rects(all_cores)
    worker_crs = _rects(active)

    x_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0
    elem_bytes = datum_bytes(x_dtype)
    defines = [("AGRMS_HAS_GAMMA", "1")] if has_gamma else []
    ablate = [(f"AGRMS_ABLATE_{PERF_ABLATE}", "1")] if PERF_ABLATE else []
    ablate += [("AGRMS_PROFILE_ZONES", "1")] if PROFILE_ZONES else []
    defines += [
        d for d in ablate if not (d[0] == "AGRMS_PROFILE_ZONES" and os.environ.get("AGRMS_PROFILE_PORT_ONLY") == "1")
    ]
    if PRESEND_A:
        ablate += [("AGRMS_PRESEND_A", "1")]
    if PRESEND_SPLIT_FENCE:
        ablate += [("AGRMS_PRESEND_SPLIT_FENCE", "1")]
    ablate += [("AGRMS_PRESEND_DRAM_ITEMS", str(PRESEND_DRAM_ITEMS))]
    if PRESEND_HOSTCOUNT:
        ablate += [("AGRMS_PRESEND_HOSTCOUNT", "1")]
    if PRESEND_LAZY:
        ablate += [("AGRMS_PRESEND_LAZY", "1")]
    if PRESEND_DMA_COPY:
        ablate += [("AGRMS_PRESEND_DMA_COPY", "1")]
    if PRESEND_DRAM_PUSH1:
        ablate += [("AGRMS_PRESEND_DRAM_PUSH1", "1")]
    if PRESEND_EARLY_OPEN:
        ablate += [("AGRMS_PRESEND_EARLY_OPEN", "1")]

    ready_sem = ready_semaphore(mesh_device, port_union)
    ready_addr = int(ttnn.get_global_semaphore_address(ready_sem))
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    def entries_of(sends, l):
        out = []
        for q, part in sends:
            out.append((q, *link_cols(geo.Wt_lp, L, l, part, FAR_ROW_SPLIT)))
        return out

    def has_sends(coord, d, l):
        ch = chips[coord]
        peer = ch.nxt if d == "fwd" else ch.prv
        return peer is not None and any(c1 > c0 for _, c0, c1, _ in entries_of(ch.sends[d], l))

    reader_ct = [
        T_in,
        geo.Wt_lp,
        w_pitch,
        n_groups,
        BLOCK_ROW_TILES,
        1 if geo.regime_n else 0,
        geo.W_local,
        G,
        geo.out_w_tiles,
        unit,
        elem_bytes,
        n_streams,
        SEM_LOCAL_LOADED,
        plan_.get(CB_X_ALIGNED, (0, 0))[0],
        1 if native_h else 0,  # own slice read in place from the input shard (cb_x_in)
        1 if stage else 0,  # regime N bfp8: compose from the bf16 staging row (cb_x_stage)
    ] + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    owner = 1 if split.combine_owner else 0
    row_gather = _div_up(k, ROW_GATHER)
    compute_ct = [k, w_pitch, BLOCK_ROW_TILES, 1 if geo.regime_n else 0, unit, geo.out_w_tiles, owner, row_gather]
    row_partials = 1 if row_partials_on(split) else 0
    compute_ct += [1 if native_h else 0, geo.Wt_lp, 1 if stage else 0, 1 if prescale else 0, row_partials]
    writer_ct = [
        out_page,
        geo.out_w_tiles,
        k,
        unit,
        plan_[CB_OUT][0] if native is None else 0,
        SEM_PARTIALS,
        1 if PARTIAL_FACES_ONLY else 0,
        1 if (has_gamma and gamma.layout == ttnn.TILE_LAYOUT) else 0,
        datum_bytes(gamma.dtype) if has_gamma else 0,
        T_g,
        g_tail,
        g_page,
        1 if geo.regime_n else 0,
        BLOCK_ROW_TILES,
        n_groups,
        owner,
        row_gather,
        SEM_INV,
        plan_.get(CB_GATHERED, (0, 0))[0],
        SEM_PARTIALS_ODD,
        GAMMA_READ_ALIGN,
        1 if native is not None else 0,  # output packed into the resident output shard (no NoC writes)
        row_partials,
    ]
    writer_ct += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct += list(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    port_ct = [T_in, run_pages, group, geo.R_t, n_groups, k, w_pitch, xfer_rows, geo.Wt_lp, RELAY_LAG]

    inv_w_bits = _f32_bits(1.0 / geo.W)
    eps_bits = _f32_bits(epsilon)
    cb_dtype = {
        CB_X_LANDING: x_dtype,
        CB_GAMMA: gamma.dtype if has_gamma else x_dtype,
        CB_ONES: ttnn.bfloat16,
        CB_SUMSQ: t_acc,
        CB_PARTIAL: t_acc,
        CB_GATHERED: t_acc,
        CB_INV_RMS: t_acc,
        CB_INV_OWNED: t_acc,
        CB_X_ALIGNED: ttnn.bfloat16 if stage else x_dtype,
        CB_X_STAGE: ttnn.bfloat16,
        CB_X_GAMMA: t_acc,
        CB_OUT: x_dtype,
        CB_X_IN: x_dtype,
    }
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)

    hash_key = (
        "all_gather_rms_norm",
        PERF_ABLATE,
        PROFILE_ZONES,
        FORCE_K,
        NATIVE_SHARD,
        PRESCALE_GAMMA,
        BLOCK_ROW_TILES,
        OUT_UNIT_CAP,
        K_SYM_MAX,
        ROW_PARTIALS,
        RELAY_LAG,
        ITEM_MIN_CHUNKS,
        FAR_ROW_SPLIT,
        PORT_CB_BYTES,
        PRESEND_A,
        PRESEND_DRAM_ITEMS,
        PRESEND_SPLIT_FENCE,
        PRESEND_HOSTCOUNT,
        PRESEND_LAZY,
        PRESEND_EARLY_OPEN,
        PRESEND_DMA_COPY,
        PRESEND_DRAM_PUSH1,
        PRESEND_EARLY_ITEMS,
        tuple(input_tensor.shape),
        str(x_dtype),
        str(gamma.dtype) if has_gamma else None,
        str(gamma.layout) if has_gamma else None,
        str(input_tensor.memory_config()),
        str(output_tensor.memory_config()),
        cluster_axis,
        ring,
        L,
        fp32_dest,
        str(compute_kernel_config.math_fidelity),
        bool(compute_kernel_config.math_approx_mode),
        bool(getattr(compute_kernel_config, "dst_full_sync_en", False)),
        k,
        n_groups,
        xfer_rows,
        native.kind if native is not None else None,
    )

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch.p
        port_cores = [ch.ports[(d, l)] for d in DIRS for l in range(L)]
        port_crs = _rects(port_cores)

        # CB 0 first, on workers + every chip's port cores: its base address is identical everywhere
        cbs = [_cb(CB_X_LANDING, *plan_[CB_X_LANDING], cb_dtype[CB_X_LANDING], all_crs)]
        for idx, (pages, page_bytes) in plan_.items():
            if idx != CB_X_LANDING:
                cbs.append(_cb(idx, pages, page_bytes, cb_dtype[idx], worker_crs))
        cbs.append(_cb(CB_PORT_CHUNKS, 2 * group * run_pages, T_in, x_dtype, port_crs))
        if native is not None:  # zero-copy: the shards ARE these CBs' L1
            cbs.append(_shard_backed_cb(CB_OUT, output_tensor, worker_crs))
            if native_h:
                cbs.append(_shard_backed_cb(CB_X_IN, input_tensor, worker_crs))
        semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(NUM_SEMS)]
        program = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)

        # ---- per-stream flags: 1 if the stream arriving at port (d, l) touches member m (any entry, any row) ----
        stream_cnt = {}
        for d in DIRS:
            up = ch.prv if d == "fwd" else ch.nxt
            for l in range(L):
                ents = entries_of(chips[up].sends[d], l) if up is not None else []
                ents = [e for e in ents if e[2] > e[1]]
                cnt = []
                for m in range(k):
                    n = 0
                    for q, c0, c1, _ in ents:
                        a, b = max(q * geo.Wt_lp + c0, ws[m]), min(q * geo.Wt_lp + c1, ws[m + 1])
                        n += a < b
                    cnt.append(min(n, 1))
                stream_cnt[(d, l)] = (ents, cnt)
        # coverage check: every landing tile of every member is written exactly once (own load or one stream), on an
        # even and an odd tile-row (row-filtered entries)
        for m in range(k):
            for par in (0, 1):
                cover = [0] * (ws[m + 1] - ws[m])
                for t in range(ws[m], ws[m + 1]):
                    if p * geo.Wt_lp <= t < (p + 1) * geo.Wt_lp:
                        cover[t - ws[m]] += 1
                for ents, _ in stream_cnt.values():
                    for q, c0, c1, rf in ents:
                        if not entry_has_row(rf, par):
                            continue
                        for t in range(max(q * geo.Wt_lp + c0, ws[m]), min(q * geo.Wt_lp + c1, ws[m + 1])):
                            cover[t - ws[m]] += 1
                assert all(
                    c == 1 for c in cover
                ), f"all_gather_rms_norm: landing coverage {cover} (chip {coord}, m {m}, row parity {par})"

        # ---- member kernels ----
        port_xy = [(lambda v: v.x | (v.y << 16))(virt(c)) for c in port_cores]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        own_members = 0
        for gi in range(n_groups):
            n_rows = group_rows[gi]
            for m in range(k):
                x, y = active[gi * k + m]
                wst, cw = ws[m], ws[m + 1] - ws[m]
                if max(wst, p * geo.Wt_lp) < min(wst + cw, (p + 1) * geo.Wt_lp):
                    own_members += 1
                streams = [stream_cnt[(d, l)][1][m] for d in DIRS for l in range(L)]
                early = SEM_EARLY + gi // xfer_rows if gi // xfer_rows < n_early else 0xFFFFFFFF
                reader_rt[x][y] = [x_addr, gi, n_rows, wst, cw, p, len(port_xy)] + port_xy + streams + [early]
                # native: a ragged last WIDTH member pads each output row up to the shard pitch
                t_cnt = geo.out_w_tiles if geo.regime_n else cw
                out_pad = native.out_pitch - t_cnt if native is not None else 0
                compute_rt[x][y] = [n_rows, cw, inv_w_bits, eps_bits, t_cnt + g_tail, m, p * geo.Wt_lp, out_pad]
                peers = [noc_xy[gi * k + mm] for mm in range(k)]
                writer_rt[x][y] = [out_addr, gi, n_rows, wst, cw, m, g_addr, out_pad] + peers
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_reader.cpp"),
                core_ranges=worker_crs,
                compile_time_args=reader_ct,
                defines=defines,
                runtime_args=reader_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_writer.cpp"),
                core_ranges=worker_crs,
                compile_time_args=writer_ct,
                defines=defines,
                runtime_args=writer_rt,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_compute.cpp"),
                core_ranges=worker_crs,
                compile_time_args=compute_ct,
                defines=defines,
                runtime_args=compute_rt,
                config=compute_kernel_config,
            ),
        ]

        # members with own columns in the groups of early item j
        early_expected = [0] * n_early
        for gi in range(n_groups):
            if gi // xfer_rows < n_early:
                for m in range(k):
                    if max(ws[m], p * geo.Wt_lp) < min(ws[m + 1], (p + 1) * geo.Wt_lp):
                        early_expected[gi // xfer_rows] += 1
        # ---- port kernels ----
        preader_rt, psender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for d in DIRS:
            peer = ch.nxt if d == "fwd" else ch.prv
            for l in range(L):
                x, y = ch.ports[(d, l)]
                out_ents = entries_of(ch.sends[d], l) if peer is not None else []
                in_ents, cnt = stream_cnt[(d, l)]
                up = ch.prv if d == "fwd" else ch.nxt
                in_all = entries_of(chips[up].sends[d], l) if up is not None else []
                s_id = DIRS.index(d) * L + l
                assert not native_h or k == 1  # the walk's from_input source assumes one member per row
                seg_args = _segment_args(out_ents, ws, geo.Wt_lp)
                preader_rt[x][y] = (
                    [own_members, s_id, len(out_ents)]
                    + [_pack_entry(*e) for e in out_ents]
                    + seg_args
                    + [len(in_all)]
                    + [_pack_entry(*e) for e in in_all]
                    + cnt
                    + [x_addr]
                    + [n_early, SEM_EARLY]
                    + [early_expected[j] for j in range(n_early)]
                )
                sending = has_sends(coord, d, l)
                # the peer's opposite-direction port (same link) sends into me: it waits for my ready
                send_ready = peer is not None and has_sends(peer, _OPP[d], l)
                args = [len(out_ents)] + [_pack_entry(*e) for e in out_ents] + seg_args
                # presend: the sender's total chunk count (kernel walk: rows_in(0, R_t) * chunks_per_row per entry)
                n_chunks = 0
                for q_, c0_, c1_, rf_ in out_ents:
                    if c1_ <= c0_:
                        continue
                    rows_ = geo.R_t if rf_ == 0 else ((geo.R_t + 1) // 2 if rf_ == 1 else geo.R_t // 2)
                    t0_, t1_ = q_ * geo.Wt_lp + c0_, q_ * geo.Wt_lp + c1_
                    cpr_ = sum(
                        _div_up(min(t1_, ws[m + 1]) - max(t0_, ws[m]), run_pages)
                        for m in range(k)
                        if max(t0_, ws[m]) < min(t1_, ws[m + 1])
                    )
                    n_chunks += rows_ * cpr_
                args += [n_chunks]
                if peer is not None and (sending or send_ready):
                    rc = virt(chips[peer].ports[(_OPP[d], l)])
                    pc = virt(chips[peer].ports[(d, l)])
                    pn = node(peer)
                    args += [ready_addr, 1 if send_ready else 0, rc.x, rc.y, pc.x, pc.y]
                    args += [int(pn.mesh_id), int(pn.chip_id), 1]
                    args += list(
                        ttnn.setup_fabric_connection(node(coord), pn, ch.links[d][l], program, ttnn.CoreCoord(x, y))
                    )
                else:
                    args += [ready_addr, 0, 0, 0, 0, 0, 0, 0, 0]
                psender_rt[x][y] = args
        reader_port_ct = port_ct + [SEM_ARRIVAL, SEM_LOCAL_LOADED, rows_per_group, 1 if native_h else 0]
        reader_port_ct += list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        sender_port_ct = port_ct + [SEM_ARRIVAL, rows_per_group]
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_reader.cpp"),
                core_ranges=port_crs,
                compile_time_args=reader_port_ct,
                defines=ablate,
                runtime_args=preader_rt,
                common_runtime_args=noc_xy,
                config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_sender.cpp"),
                core_ranges=port_crs,
                compile_time_args=sender_port_ct,
                defines=ablate,
                runtime_args=psender_rt,
                common_runtime_args=noc_xy,
                config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
            ),
        ]
        program.kernels = kernels
        program.custom_program_hash = hash((hash_key, coord)) & ((1 << 64) - 1)
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
