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

# ---- Blocking-model knobs (op_design.md -> Blocking Model). Every extent below derives from these. ----
BLOCK_ROW_TILES = 1  # tile-rows per block (rows land progressively: one row per block)
OUT_UNIT_CAP = 16  # writer / compose unit (tiles)
OUT_DEPTH = 2  # cb_out_tiles depth in units
COMPOSE_DEPTH = 2  # cb_x_aligned depth in units (regime N)
ITEM_MIN_CHUNKS = int(
    os.environ.get("AGRMS_ITEM_MIN_CHUNKS", "4")
)  # (env: perf probes) chunks per transport item (one fused write + increment per item)
PORT_CB_BYTES = 112 * 1024  # cb_port_chunks bytes (2 batches)
PORT_GROUP_CAP = 8  # chunks per port batch
RELAY_LAG = int(
    os.environ.get("AGRMS_RELAY_LAG", "2")
)  # (env: perf probes) steps between a relay item's upstream send and its relay (>= 1)
K_SYM_MAX = int(
    os.environ.get("AGRMS_K_SYM_MAX", "8")
)  # (env: perf probes) symmetric all-gather of partials up to this k; per-row owner combine beyond (base op)
ROW_GATHER = 32  # owner combine: members per row-form gather tile (one 32-datum row each)
OCC_TILES_PER_MEMBER = 4  # member width floor (except the smallest candidate)
COMBINE_TILE_COST = 3  # per tile-row combine cost (x-tile equivalents), base op
OWNER_MEMBER_COST = 0.4  # owner combine: per member (x-tile equivalents), base op
TILIZED_WTILE_NS = 310  # per W tile of a tile-row, base op
KEEP_UP_FRACTION = 0.5
XFER_FIT = {True: (8.4e-6, 77.2e9), False: (9.8e-6, 42.1e9)}  # (fixed s, bytes/s per link): fabric_all_gather fits
PARTIAL_FACES_ONLY = True
GAMMA_READ_ALIGN = 64
MAX_RUNTIME_ARGS = 341
# Perf probes only (never set by the op): AGRMS_ABLATE=EMPTY | LOCAL | TRANSPORT stubs one side of the program
# (kernels/*: AGRMS_ABLATE_*). EMPTY: every kernel returns at once (launch floor). LOCAL: no transport (ports idle,
# members do not wait for landings). TRANSPORT: members only load, signal and wait for their landings.
PERF_ABLATE = os.environ.get("AGRMS_ABLATE", "")
# Test hook only (debug tests pin a split): AGRMS_FORCE_K = k restricts select_split to that candidate.
FORCE_K = int(os.environ.get("AGRMS_FORCE_K", "0"))


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


def link_cols(Wt_lp, L, l, part):
    """Columns [c0, c1) of a slice that link l carries for an entry part (part halves nest in the link's range)."""
    a, b = (l * Wt_lp) // L, ((l + 1) * Wt_lp) // L
    if part == 0:
        return a, b
    h = a + (b - a) // 2
    return (a, h) if part == 1 else (h, b)


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


def plan(mesh_device, cluster_axis, ring, L):
    key = (id(mesh_device), cluster_axis, ring, L)
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
    allowed = allowed_cores(mesh_device)
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


def cb_plan(geo: Geometry, sz: Sizes, split: Split):
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
            plan_[CB_GATHERED] = (depth_g * k * b, sz.T_acc)
    if geo.regime_n:
        plan_[CB_X_ALIGNED] = (COMPOSE_DEPTH * unit, sz.T_in)
    return plan_


def _plan_bytes(p):
    return sum(pages * page for pages, page in p.values())


def port_footprint(sz: Sizes, split: Split):
    return split.blocks_max * BLOCK_ROW_TILES * split.w_pitch * sz.T_in + 2 * sz.group * sz.run_pages * sz.T_in


def select_split(geo: Geometry, sz: Sizes, n_workers: int, budget: int, ring: bool, L: int):
    """op_design.md -> Regime / split selection. Objective: the tail after the last landing, among the splits that
    keep up with the transport; else the smallest total; else RuntimeError."""
    S = geo.M * geo.num_images * geo.W_local * sz.T_in / (TILE * TILE)
    t0, rate = XFER_FIT[ring]
    T_xfer = t0 + (geo.G - 1) * S / (rate * L)
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
        C = 0 if k == 1 else (COMBINE_TILE_COST if not owner else COMBINE_TILE_COST + OWNER_MEMBER_COST * k)
        total = rows_max * (w_pitch + C)
        tail = BLOCK_ROW_TILES * (w_pitch + C)
        fits.append((total, rows_max, split))
        if total * TILIZED_WTILE_NS * 1e-9 <= KEEP_UP_FRACTION * T_xfer:
            keeps.append((tail, rows_max, split))
    if keeps:
        return min(keeps, key=lambda t: (t[0], t[1]))[2]
    if fits:
        return min(fits, key=lambda t: (t[0], t[1]))[2]
    raise RuntimeError(
        f"all_gather_rms_norm: no admissible split fits L1 (budget {budget} B) for per-device shape "
        f"rows {geo.R_t} x slice {geo.Wt_lp} tiles x G {geo.G} (dram_landing regime is deferred)"
    )


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


def _pack_entry(q, c0, c1):
    assert q < 256 and c0 < 4096 and c1 < 4096
    return (q << 24) | (c0 << 12) | c1


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
    chips, G = plan(mesh_device, cluster_axis, ring, L)
    geo = derive_geometry(input_tensor.shape, G)
    has_gamma = gamma is not None
    x_dtype = input_tensor.dtype
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    t_acc = acc_dtype(fp32_dest)
    T_in = ttnn.tile_size(x_dtype)
    run_pages = max(1, int(ttnn.get_tt_fabric_max_payload_size_bytes()) // T_in)
    group = max(1, min(PORT_GROUP_CAP, PORT_CB_BYTES // (2 * run_pages * T_in)))
    T_g, g_tail = gamma_page_layout(gamma.dtype) if has_gamma else (0, 0)
    sz = Sizes(T_in, ttnn.tile_size(t_acc), T_g, g_tail, run_pages, group, has_gamma)

    # ---- worker set: allowed cores minus the union of every chip's port cores (identical on every chip) ----
    allowed = allowed_cores(mesh_device)
    port_union = {c for ch in chips.values() for c in ch.ports.values()}
    workers = sorted(allowed - port_union, key=lambda c: (c[1], c[0]))
    if not workers:
        raise ValueError("all_gather_rms_norm: no worker core left after the ports")
    split = select_split(geo, sz, len(workers), cb_l1_budget(mesh_device), ring, L)
    k, ws, w_pitch, n_groups = split.k, split.ws, split.w_pitch, split.n_groups
    active = workers[: n_groups * k]  # worker index g * k + m
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

    cpr = max(cpr_of(0, *link_cols(geo.Wt_lp, L, l, 0)) for l in range(L))
    xfer_rows = max(1, _div_up(ITEM_MIN_CHUNKS, max(cpr, 1)))

    out_w = geo.out_w_tiles if geo.regime_n else w_pitch  # tiles normalized / stored per row (pitch)
    unit = out_unit_of(geo, out_w)
    plan_ = cb_plan(geo, sz, split)

    # ---- semaphores (program, on every core of every chip's union set; ids identical everywhere) ----
    n_streams = 2 * L
    SEM_PARTIALS = n_streams
    SEM_LOCAL_LOADED = n_streams + 1
    SEM_ARRIVAL = n_streams + 2
    SEM_INV = n_streams + 3  # owner combine: inv_rms returns

    all_cores = sorted(set(workers) | port_union, key=lambda c: (c[1], c[0]))
    all_crs = _rects(all_cores)
    worker_crs = _rects(active)

    x_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0
    elem_bytes = datum_bytes(x_dtype)
    defines = [("AGRMS_HAS_GAMMA", "1")] if has_gamma else []
    ablate = [(f"AGRMS_ABLATE_{PERF_ABLATE}", "1")] if PERF_ABLATE else []
    defines += ablate

    ready_sem = ready_semaphore(mesh_device, port_union)
    ready_addr = int(ttnn.get_global_semaphore_address(ready_sem))
    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    def entries_of(sends, l):
        out = []
        for q, part in sends:
            c0, c1 = link_cols(geo.Wt_lp, L, l, part)
            out.append((q, c0, c1))
        return out

    def has_sends(coord, d, l):
        ch = chips[coord]
        peer = ch.nxt if d == "fwd" else ch.prv
        return peer is not None and any(c1 > c0 for _, c0, c1 in entries_of(ch.sends[d], l))

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
    ] + list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    owner = 1 if split.combine_owner else 0
    row_gather = _div_up(k, ROW_GATHER)
    compute_ct = [k, w_pitch, BLOCK_ROW_TILES, 1 if geo.regime_n else 0, unit, geo.out_w_tiles, owner, row_gather]
    writer_ct = [
        out_page,
        geo.out_w_tiles,
        k,
        unit,
        plan_[CB_OUT][0],
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
        CB_X_ALIGNED: x_dtype,
        CB_OUT: x_dtype,
    }
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(processor=risc, noc=noc)

    hash_key = (
        "all_gather_rms_norm",
        PERF_ABLATE,
        FORCE_K,
        K_SYM_MAX,
        RELAY_LAG,
        ITEM_MIN_CHUNKS,
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
        k,
        n_groups,
        xfer_rows,
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
        semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(SEM_INV + 1)]
        program = ttnn.ProgramDescriptor(kernels=[], semaphores=semaphores, cbs=cbs)

        # ---- per-stream counts: entries of the stream arriving at port (d, l) that touch member m ----
        stream_cnt = {}
        for d in DIRS:
            up = ch.prv if d == "fwd" else ch.nxt
            for l in range(L):
                ents = entries_of(chips[up].sends[d], l) if up is not None else []
                ents = [e for e in ents if e[2] > e[1]]
                cnt = []
                for m in range(k):
                    n = 0
                    for q, c0, c1 in ents:
                        a, b = max(q * geo.Wt_lp + c0, ws[m]), min(q * geo.Wt_lp + c1, ws[m + 1])
                        n += a < b
                    cnt.append(n)
                stream_cnt[(d, l)] = (ents, cnt)
        # coverage check: every landing tile of every member is written exactly once (own load or one stream)
        for m in range(k):
            cover = [0] * (ws[m + 1] - ws[m])
            for t in range(ws[m], ws[m + 1]):
                if p * geo.Wt_lp <= t < (p + 1) * geo.Wt_lp:
                    cover[t - ws[m]] += 1
            for ents, _ in stream_cnt.values():
                for q, c0, c1 in ents:
                    for t in range(max(q * geo.Wt_lp + c0, ws[m]), min(q * geo.Wt_lp + c1, ws[m + 1])):
                        cover[t - ws[m]] += 1
            assert all(c == 1 for c in cover), f"all_gather_rms_norm: landing coverage {cover} (chip {coord}, m {m})"

        # ---- member kernels ----
        port_xy = [(lambda v: v.x | (v.y << 16))(virt(c)) for c in port_cores]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        own_members = 0
        for gi in range(n_groups):
            n_rows = len(range(gi, geo.R_t, n_groups))
            for m in range(k):
                x, y = active[gi * k + m]
                wst, cw = ws[m], ws[m + 1] - ws[m]
                if max(wst, p * geo.Wt_lp) < min(wst + cw, (p + 1) * geo.Wt_lp):
                    own_members += 1
                streams = [stream_cnt[(d, l)][1][m] for d in DIRS for l in range(L)]
                reader_rt[x][y] = [x_addr, gi, n_rows, wst, cw, p, len(port_xy)] + port_xy + streams
                compute_rt[x][y] = [n_rows, cw, inv_w_bits, eps_bits, (out_w if geo.regime_n else cw) + g_tail, m]
                peers = [noc_xy[gi * k + mm] for mm in range(k)]
                writer_rt[x][y] = [out_addr, gi, n_rows, wst, cw, m, g_addr] + peers
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
                preader_rt[x][y] = (
                    [own_members, s_id, len(out_ents)]
                    + [_pack_entry(*e) for e in out_ents]
                    + [len(in_all)]
                    + [_pack_entry(*e) for e in in_all]
                    + cnt
                    + ws
                )
                sending = has_sends(coord, d, l)
                # the peer's opposite-direction port (same link) sends into me: it waits for my ready
                send_ready = peer is not None and has_sends(peer, _OPP[d], l)
                args = [len(out_ents)] + [_pack_entry(*e) for e in out_ents] + ws
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
        reader_port_ct = port_ct + [SEM_ARRIVAL, SEM_LOCAL_LOADED]
        sender_port_ct = port_ct + [SEM_ARRIVAL]
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
