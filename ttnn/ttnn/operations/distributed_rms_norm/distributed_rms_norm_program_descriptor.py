# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm MeshProgramDescriptor (op_design.md -> Blocking Model, Work Distribution, Dataflow Strategy).

One program per chip, one ttnn.generic_op per call. Per chip:

  port cores (2 * L, one per (direction, link))   the fabric_ring_exchange port kernel (relay all-gather of the
                                                  compact fp32 Σx² records, ready fence, per-entry notify)
  worker cores (grid - ports)                     reader / compute / writer — the base op's local work, re-split
                                                  for the worker grid and cut into row WAVES:
      Phase A  load x, square + collapse per wave, on-chip combine of the G_w slice partials at the wave's owner,
               owner publishes the wave's compact records into its link's ports;  γ pass over the whole block
      Phase B  owner collects the G records of each owned row (wave order), folds them in group-position order
               (SFPU, fp32), finalizes rsqrt(ss/W + eps), all-gathers inv_rms to the row-group;
               every member scales its wave rows and stores them.

Regimes (select_plan): D1 (interleaved, whole rows per core), D3 (interleaved, W split over G_w slice cores),
D1-HS (HEIGHT_SHARDED, the shard is the block), D3-WS (WIDTH_SHARDED, one row-group of all shard cores).

Knobs (single source of truth; every CB size / trip count derives from them):
    W_CHUNK_TILES      cap of the streaming chunk C (scale_by_inv_rms -> writer quantum)
    STREAM_DEPTH       cb_output depth in C-tile quanta
    SCALE_BLOCK_TILES  tiles per DEST window of the scaling chains (clamped to DEST by the chain)
    MIN_SLICE_TILES    smallest interleaved W-slice S (fused_split)
    MAX_GROUP_CORES    cap on G_w (on-chip combine fan-in)
    WAVES_MAX          cap on row waves per core (exchange pipelining depth)
    READ_LOOKAHEAD_WAVES  x-read waves in flight per core beyond the oldest unlanded one
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
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
MIN_SLICE_TILES = 4
MAX_GROUP_CORES = 24
WAVES_MAX = 8
READ_LOOKAHEAD_WAVES = 1  # x-read waves in flight beyond the oldest unlanded one (wave-ordered DRAM traffic)

# ---- Exchange constants ----
RECORD_BYTES = 32 * 4  # one fp32 per row of a tile-row
SEG_ALIGN = 64  # landing slot alignment (Blackhole DRAM/L1 NoC alignment)
MAX_HEADERS = 8  # packet headers per port (one per destination chip)
FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}

SCALER_FORMAT = ttnn.bfloat16
STATS_FORMAT = ttnn.float32  # cb_stats / cb_record_scratch: the exchanged records are exact fp32 (wire format)

# ---- CB indices (the kernels declare the same slots) ----
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
CB_CHIP_PARTIAL = 14
CB_STATS = 15
CB_OUTPUT = 16
CB_RECORD_SCRATCH = 17
NUM_CBS = 64


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _align_up(v, a):
    return (v + a - 1) // a * a


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def intermediate_format(fp32_dest_acc_en: bool):
    """Compute-internal CB format follows the DEST width (base op intermediate_format)."""
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def tensor_tile_rows_of(shape) -> int:
    return _prod(shape[:-2]) * math.ceil(shape[-2] / TILE_DIM)


def l1_budget(device) -> int:
    try:
        return int(ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank)
    except Exception:  # noqa: BLE001 — mesh devices without a memory view: Blackhole worker L1 minus reserves
        return 1300 * 1024


# ---------------------------------------------------------------------------------------------------------------
# Local plan (the base op's blocking, re-split for the worker grid left after the ports)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    core: object  # logical CoreCoord
    group: int  # row-group index
    index: int  # member index j within the row-group
    start_row: int  # first tile-row of the group
    rows: int  # tile-rows of the group
    w0: int  # first tile of the slice
    V: int  # valid tiles of the slice


@dataclass
class LocalPlan:
    regime: str
    sharded: bool
    S: int  # slice tiles (nominal)
    C: int  # chunk
    nch: int  # chunks per slice
    WS: int  # x / output row stride (tiles): Wp (interleaved) or S (shard)
    G_w: int
    G_h: int
    rows_max: int
    num_waves: int
    q_w: int
    owners: int
    q: int
    masked: bool
    members: list = field(default_factory=list)  # Member per worker core, in group order
    gamma_per_wave: bool = False  # L1 fallback: the γ pass per wave (cb_x_gamma holds q_w rows)

    @property
    def Wp(self):
        return self.nch * self.C


def gather_field_bits(G_w: int) -> int:
    """Bits per owned-wave field of SEM_GATHER (counts up to G_w contributions)."""
    return max(1, G_w.bit_length())


def max_owned_waves(num_waves: int, owners: int, G_w: int) -> int:
    """Most waves one member owns (owner k of wave w = member (w * owners + k) mod G_w)."""
    return math.ceil(num_waves * owners / G_w) if G_w > 1 else 0


def wave_params(rows_max: int, G_w: int):
    """(num_waves, q_w, owners, q). Waves shrink until every member's owned waves fit SEM_GATHER's 32 bits."""
    waves = max(1, min(rows_max, WAVES_MAX))
    while True:
        q_w = math.ceil(rows_max / waves)
        num_waves = math.ceil(rows_max / q_w)
        owners = max(1, min(G_w, q_w))
        q = math.ceil(q_w / owners)
        if waves == 1 or max_owned_waves(num_waves, owners, G_w) * gather_field_bits(G_w) <= 32:
            return num_waves, q_w, owners, q
        waves -= 1


def chunking(S: int, sharded: bool):
    nch = math.ceil(S / W_CHUNK_TILES)
    C = math.ceil(S / nch)
    if sharded:
        nch = math.ceil(S / C)
    Wp = nch * C
    WS = S if sharded else Wp
    return C, nch, WS


@dataclass(frozen=True)
class Sizes:
    T_in: int
    T_out: int
    T_g: int  # 0 when absent
    T_i: int
    has_gamma: bool
    G: int


def footprint(
    sz: Sizes, sharded, masked, S, R, G_w, num_waves, q_w, owners, q, out_sharded=None, gamma_per_wave=False
) -> int:
    """Per-worker L1 bytes of every CB (l1_ledger.md -> Total per-core footprint)."""
    C, nch, WS = chunking(S, sharded)
    Wp = nch * C
    max_owned = max_owned_waves(num_waves, owners, G_w)
    f = 0
    f += 0 if sharded else R * Wp * sz.T_in  # cb_x_tiles
    f += (2 if masked else 1) * ttnn.tile_size(SCALER_FORMAT)
    f += sz.T_i * (q_w + num_waves * q_w + 2 * q + q)  # sumsq, inv, chip, root
    if G_w > 1:  # the on-chip combine CBs exist only on D3 / D3-WS (create_mesh_program_descriptor)
        f += sz.T_i * (2 * q_w + max(1, max_owned) * G_w * q)  # partial_col, gather
    f += ttnn.tile_size(STATS_FORMAT) * sz.G * q  # stats
    f += (sz.G + 1) * q * RECORD_BYTES  # record scratch
    if sz.has_gamma:
        f += Wp * sz.T_g + (q_w if gamma_per_wave else R) * WS * sz.T_i  # cb_gamma_tiles + cb_x_gamma
    if masked:
        f += 2 * q_w * sz.T_i
    out_sharded = sharded if out_sharded is None else out_sharded
    f += 0 if out_sharded else STREAM_DEPTH * C * sz.T_out
    return f


def fused_split(T: int, Wt: int, N: int, sz: Sizes, budget: int):
    """(S, G_w, G_h) for an interleaved input on N worker cores (op_design.md -> Work Distribution)."""
    best = None
    for S in range(min(Wt, MIN_SLICE_TILES), Wt + 1):
        G_w = math.ceil(Wt / S)
        if G_w > MAX_GROUP_CORES or G_w > N:
            continue
        if S > 1 and math.ceil(Wt / (S - 1)) == G_w and S - 1 >= min(Wt, MIN_SLICE_TILES):
            continue  # a narrower slice gives the same G_w: never better
        G_h = min(T, N // G_w)
        rows_max = math.ceil(T / G_h)
        nw, q_w, owners, q = wave_params(rows_max, G_w)
        # feasibility with the per-wave γ fallback (the smaller cb_x_gamma); local_plan prefers whole-block γ
        if footprint(sz, False, False, S, rows_max, G_w, nw, q_w, owners, q, gamma_per_wave=True) > budget:
            continue
        key = (rows_max * S, -min(rows_max, WAVES_MAX), G_w)
        if best is None or key < best[0]:
            best = (key, S, G_w, G_h)
    if best is None:
        raise NotImplementedError(
            f"distributed_rms_norm: no W-split of Wt={Wt} fits L1 with the whole per-core block resident "
            f"(budget {budget} B); the x_depth = 2 fallback is not built"
        )
    return best[1], best[2], best[3]


def _shard_cores(spec, n):
    return list(ttnn.corerange_to_cores(spec.grid, n, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))


def local_plan(input_tensor, workers, sz: Sizes, budget: int, out_sharded=None) -> LocalPlan:
    shape = list(input_tensor.shape)
    W = shape[-1]
    Wt = math.ceil(W / TILE_DIM)
    T = tensor_tile_rows_of(shape)
    mc = input_tensor.memory_config()
    ml = mc.memory_layout

    if ml == ttnn.TensorMemoryLayout.INTERLEAVED:
        S, G_w, G_h = fused_split(T, Wt, len(workers), sz, budget)
        base, extra = divmod(T, G_h)
        members = []
        start = 0
        for g in range(G_h):
            rows = base + (1 if g < extra else 0)
            for j in range(G_w):
                w0 = j * S
                members.append(Member(workers[g * G_w + j], g, j, start, rows, w0, min(S, Wt - w0)))
            start += rows
        rows_max = base + (1 if extra else 0)
        regime = "D3" if G_w > 1 else "D1"
        sharded = False
        masked = False
    else:
        spec = mc.shard_spec
        if spec is None or mc.buffer_type != ttnn.BufferType.L1:
            raise NotImplementedError("distributed_rms_norm: sharded inputs must carry an L1 shard spec")
        shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
        if shard_h % TILE_DIM or shard_w % TILE_DIM:
            raise NotImplementedError("distributed_rms_norm: TILE shards must be tile multiples")
        Hsh = shard_h // TILE_DIM
        S = shard_w // TILE_DIM
        sharded = True
        masked = W % TILE_DIM != 0
        members = []
        if ml == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            if shard_w < W:
                raise NotImplementedError("distributed_rms_norm: height shard does not cover the row")
            n = math.ceil(T / Hsh)
            cores = _shard_cores(spec, n)
            for i in range(n):
                rows = min(Hsh, T - i * Hsh)
                members.append(Member(cores[i], i, 0, i * Hsh, rows, 0, Wt))
            G_w = 1
            rows_max = max(m.rows for m in members)
            regime = "D1-HS"
        elif ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
                raise NotImplementedError("distributed_rms_norm: WIDTH_SHARDED needs ROW_MAJOR orientation")
            n = math.ceil(Wt / S)
            if Hsh < T:
                raise NotImplementedError("distributed_rms_norm: width shard does not cover the height")
            cores = _shard_cores(spec, n)
            for j in range(n):
                members.append(Member(cores[j], 0, j, 0, T, j * S, min(S, Wt - j * S)))
            G_w, rows_max = n, T
            regime = "D3-WS"
        else:
            raise NotImplementedError(f"distributed_rms_norm: memory layout {ml} is not built")
        worker_set = {(c.x, c.y) for c in workers}
        if any((m.core.x, m.core.y) not in worker_set for m in members):
            raise NotImplementedError("distributed_rms_norm: a shard sits on a port core (PORT-ON-SHARD, deferred)")
        if G_w > 1 and any(m.V <= 0 for m in members):
            raise NotImplementedError("distributed_rms_norm: a width shard with no valid tile")

    C, nch, WS = chunking(S, sharded)
    num_waves, q_w, owners, q = wave_params(rows_max, G_w)
    plan = LocalPlan(
        regime=regime,
        sharded=sharded,
        S=S,
        C=C,
        nch=nch,
        WS=WS,
        G_w=G_w,
        G_h=max(m.group for m in members) + 1,
        rows_max=rows_max,
        num_waves=num_waves,
        q_w=q_w,
        owners=owners,
        q=q,
        masked=masked,
        members=members,
    )
    fp = footprint(sz, sharded, masked, S, rows_max, G_w, num_waves, q_w, owners, q, out_sharded)
    if fp > budget and sz.has_gamma:
        plan.gamma_per_wave = True  # lamp L4 fallback: x*γ per wave (cb_x_gamma = q_w rows) instead of per block
        fp = footprint(sz, sharded, masked, S, rows_max, G_w, num_waves, q_w, owners, q, out_sharded, True)
    if fp > budget:
        raise NotImplementedError(f"distributed_rms_norm: per-core footprint {fp} B exceeds L1 budget {budget} B")
    return plan


def member_of(plan: LocalPlan, w: int, k: int) -> int:
    """Member index of owner k of wave w (rotated so consecutive waves land on different members)."""
    return (w * plan.owners + k) % plan.G_w


def n_wave_rows(plan: LocalPlan, rows: int, w: int) -> int:
    r0 = w * plan.q_w
    return 0 if r0 >= rows else min(plan.q_w, rows - r0)


# ---------------------------------------------------------------------------------------------------------------
# Exchange plan (TRANSPORT: fabric_ring_exchange relay, generalized to Linear)
# ---------------------------------------------------------------------------------------------------------------


def exchange_plan(G: int, ring: bool):
    """Per position s: sends[d] = [(slot q, dest pos, dest entry, hops, wait)], entries[d] = [slots landing on port d],
    ready_out[d] = [(target pos, target direction, hops)]. Relay: item 0 = own slot, item i >= 1 relays entry i-1."""
    info = [dict(sends={FWD: [], BWD: []}, entries={FWD: [], BWD: []}, ready_out={FWD: [], BWD: []}) for _ in range(G)]
    for s in range(G):
        if ring:
            kf = G // 2
            kb = G - 1 - kf
            fwd = [(s - i) % G for i in range(kf)]
            bwd = [(s + i) % G for i in range(kb)]
            nxt, prv = (s + 1) % G, (s - 1) % G
        else:
            fwd = [s - i for i in range(s + 1)] if s < G - 1 else []
            bwd = [s + i for i in range(G - s)] if s > 0 else []
            nxt = s + 1 if s < G - 1 else None
            prv = s - 1 if s > 0 else None
        for d, lst, peer in ((FWD, fwd, nxt), (BWD, bwd, prv)):
            if not lst:
                continue
            info[s]["sends"][d] = [(q, peer, i, 1, 0 if i == 0 else i) for i, q in enumerate(lst)]
            info[peer]["entries"][d] = list(lst)
    for s in range(G):
        for d in (FWD, BWD):
            for t in sorted({it[1] for it in info[s]["sends"][d]}):
                info[t]["ready_out"][_OPP[d]].append((s, d, 1))
    return info


def port_cores(grid, n_ports, taken=frozenset()):
    """n_ports free cores nearest the middle of logical row 0 (next to the Ethernet row), row by row."""
    cx = (grid.x - 1) / 2.0
    cand = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    cand.sort(key=lambda c: (c[1], abs(c[0] - cx), c[0]))
    if len(cand) < n_ports:
        raise NotImplementedError("distributed_rms_norm: too few free cores for the ports (PORT-ON-SHARD)")
    sel = sorted(cand[:n_ports], key=lambda c: (c[1], c[0]))
    return [ttnn.CoreCoord(x, y) for x, y in sel]


# ---------------------------------------------------------------------------------------------------------------
# Persistent exchange resources (cold call only)
# ---------------------------------------------------------------------------------------------------------------

_RESOURCES = {}


class Resources:
    def __init__(self, mesh, ports, G, num_waves, seg_stride):
        import torch  # host-side only

        self.ports = ports
        crs = _crs(ports)
        self.sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(G)]  # ready + G-1 arrivals
        words = max(1, num_waves * G * seg_stride // 4)
        n_dev = mesh.get_num_devices()
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.landing = ttnn.from_torch(
            torch.zeros((n_dev * len(ports), words), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.synchronize_device(mesh)  # every chip's counters are zero before any chip's first program

    @property
    def ready_addr(self):
        return int(ttnn.get_global_semaphore_address(self.sems[0]))

    @property
    def arrival_addrs(self):
        return [int(ttnn.get_global_semaphore_address(s)) for s in self.sems[1:]]

    @property
    def landing_addr(self):
        return int(self.landing.buffer_address())


def get_resources(mesh, ports, G, num_waves, seg_stride):
    key = (id(mesh), tuple((c.x, c.y) for c in ports), G, num_waves, seg_stride)
    res = _RESOURCES.get(key)
    if res is None:
        res = (mesh, Resources(mesh, ports, G, num_waves, seg_stride))
        _RESOURCES[key] = res
    res[1].key = key
    return res[1]


def forget_resources(res):
    """Drop cached resources whose device-side initialization never ran (see dispatch_collected)."""
    _RESOURCES.pop(getattr(res, "key", None), None)


def collected_count() -> int:
    """Ops stashed by the up-front precompile collector (tests/plugins/up_front_collect.py). That collector runs a test
    body with dispatch blocked: a cold call made there creates its global semaphores and landing scratch without the
    device ever seeing their initial values, so they must not be cached for the real run."""
    fn = getattr(getattr(ttnn, "graph", None), "up_front_num_collected", None)
    try:
        return int(fn()) if fn is not None else 0
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------------------------------------------
# Groups / links
# ---------------------------------------------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def group_hops(G, ring):
    hops = [(i, i + 1) for i in range(G - 1)]
    if ring and G >= 3:
        hops.append((G - 1, 0))
    return hops


def discovered_links(mesh, cluster_axis, ring):
    """min over every hop (both directions) of every group of the forwarding links; raises on a missing hop."""
    best = None
    for grp in build_groups(tuple(mesh.shape), cluster_axis):
        for a, b in group_hops(len(grp), ring):
            for s, d in ((grp[a], grp[b]), (grp[b], grp[a])):
                n = len(ttnn.get_forwarding_link_indices(_node(mesh, s), _node(mesh, d)))
                if n == 0:
                    raise ValueError(
                        f"distributed_rms_norm: no direct forwarding link {s} -> {d} "
                        f"(topology={'Ring' if ring else 'Linear'}, fabric={ttnn.get_fabric_config()})"
                    )
                best = n if best is None else min(best, n)
    return best or 0


# ---------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------


_ABLATE_DEFINES = {"exchange": "DRMS_ABLATE_EXCHANGE"}


def _measurement_defines() -> list:
    """Measurement-only kernel defines (never set in production: unset env => no defines, binaries unchanged).
    DRMS_ABLATE=exchange: ports return at once, workers skip the counted gate and the landed waits."""
    import os

    ablate = {a.strip() for a in os.environ.get("DRMS_ABLATE", "").split(",") if a.strip()}
    unknown = ablate - set(_ABLATE_DEFINES)
    if unknown:
        raise ValueError(f"DRMS_ABLATE: unknown switches {sorted(unknown)}")
    defines = [(d, "1") for a, d in _ABLATE_DEFINES.items() if a in ablate]
    if os.environ.get("DRMS_PERF_ZONES", "0") not in ("", "0"):
        defines.append(("DRMS_PERF_ZONES", "1"))
    return defines


def _cb(index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _dtype_kind(dtype):
    return {ttnn.float32: 0, ttnn.bfloat16: 1, ttnn.bfloat8_b: 2}[dtype]


def _gamma_es(gamma):
    return ttnn.tile_size(gamma.dtype) // TILE_HW


def plan_summary(input_tensor, *, cluster_axis, num_links, gamma=None, fp32_dest_acc_en=True):
    """Host-only: the selected regime and its split (for tests)."""
    mesh = input_tensor.device()
    grid = mesh.compute_with_storage_grid_size()
    taken = set()
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        taken = {(c.x, c.y) for r in mc.shard_spec.grid.ranges() for c in _range_cores(r)}
    ports = port_cores(grid, 2 * num_links, taken)
    pset = {(c.x, c.y) for c in ports}
    workers = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pset]
    sz = _sizes(input_tensor, gamma, fp32_dest_acc_en, mesh.shape[cluster_axis])
    p = local_plan(input_tensor, workers, sz, l1_budget(mesh))
    return dict(
        regime=p.regime,
        gamma_per_wave=p.gamma_per_wave,
        S=p.S,
        G_w=p.G_w,
        G_h=p.G_h,
        rows_max=p.rows_max,
        num_waves=p.num_waves,
        workers=len(workers),
    )


def _range_cores(r):
    return [ttnn.CoreCoord(x, y) for y in range(r.start.y, r.end.y + 1) for x in range(r.start.x, r.end.x + 1)]


def _sizes(input_tensor, gamma, fp32_dest_acc_en, G):
    interm = intermediate_format(fp32_dest_acc_en)
    return Sizes(
        T_in=ttnn.tile_size(input_tensor.dtype),
        T_out=ttnn.tile_size(input_tensor.dtype),
        T_g=ttnn.tile_size(gamma.dtype) if gamma is not None else 0,
        T_i=ttnn.tile_size(interm),
        has_gamma=gamma is not None,
        G=G,
    )


def create_mesh_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma,
    epsilon: float,
    cluster_axis: int,
    ring: bool,
    num_links: int,
    compute_kernel_config,
):
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    grid = mesh.compute_with_storage_grid_size()
    G = mesh_shape[cluster_axis]
    ring = ring and G >= 3  # a 2-chip ring is the line it physically is
    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    interm = intermediate_format(fp32_acc)
    sz = _sizes(input_tensor, gamma, fp32_acc, G)
    shape = list(input_tensor.shape)
    W_local = shape[-1]
    Wt = math.ceil(W_local / TILE_DIM)
    w_valid_last = W_local - TILE_DIM * (Wt - 1)

    # ---- ports + worker grid ----
    mc = input_tensor.memory_config()
    taken = set()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        taken = {(c.x, c.y) for r in mc.shard_spec.grid.ranges() for c in _range_cores(r)}
    port_list = port_cores(grid, 2 * num_links, taken)
    pset = {(c.x, c.y) for c in port_list}
    workers = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pset]
    out_sharded = output_tensor.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    plan = local_plan(input_tensor, workers, sz, l1_budget(mesh), out_sharded)

    # ---- publishers -> links ----
    P = plan.G_h * plan.owners  # publishers per wave
    L = min(num_links, P)
    n_pub = [len([k for k in range(P) if k % L == l]) for l in range(L)]
    seg_bytes = [n_pub[l] * plan.q * RECORD_BYTES for l in range(L)]
    seg_stride = _align_up(max(seg_bytes), SEG_ALIGN)
    ports = {(d, l): port_list[2 * l + d] for l in range(L) for d in (FWD, BWD)}
    used_ports = list(ports.values())
    res = get_resources(mesh, used_ports, G, plan.num_waves, seg_stride)
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    pack = lambda v: (v.x << 16) | v.y

    # roots of link l: every core that publishes on link l in some wave
    groups = {}
    for m in plan.members:
        groups.setdefault(m.group, {})[m.index] = m
    roots = [[] for _ in range(L)]
    for w in range(plan.num_waves):
        for g in range(plan.G_h):
            for k in range(plan.owners):
                l = (g * plan.owners + k) % L
                c = groups[g][member_of(plan, w, k)].core
                pc = pack(virt(c))
                if pc not in roots[l]:
                    roots[l].append(pc)

    xinfo = exchange_plan(G, ring)

    # ---- semaphore ids ----
    SEM_GATHER = 0
    SEM_INV = 1
    SEM_LANDED_BASE = 2  # + 2*l + d (workers: landed notifies per link and direction)
    SEM_OWN_STATS_BASE = 2 + 2 * L  # + w (ports: publishers of wave w announced)
    n_sems = SEM_OWN_STATS_BASE + plan.num_waves
    assert n_sems <= 16, "NUM_SEMAPHORES"

    worker_crs = _crs([m.core for m in plan.members])
    port_crs = _crs(used_ports)
    all_crs = _crs([m.core for m in plan.members] + used_ports)

    # ---- CBs (worker cores) ----
    R = plan.rows_max
    max_owned = max_owned_waves(plan.num_waves, plan.owners, plan.G_w)
    cbs = [
        _cb(CB_REDUCE_SCALER, SCALER_FORMAT, ttnn.tile_size(SCALER_FORMAT), 2 if plan.masked else 1, worker_crs),
        _cb(CB_SUMSQ_PARTIAL, interm, sz.T_i, plan.q_w, worker_crs),
        _cb(CB_INV_RMS, interm, sz.T_i, plan.num_waves * plan.q_w, worker_crs),
        _cb(CB_CHIP_PARTIAL, interm, sz.T_i, 2 * plan.q, worker_crs),
        _cb(CB_RMS_ROOT, interm, sz.T_i, plan.q, worker_crs),
        _cb(CB_STATS, STATS_FORMAT, ttnn.tile_size(STATS_FORMAT), G * plan.q, worker_crs),
        _cb(CB_RECORD_SCRATCH, STATS_FORMAT, RECORD_BYTES, (G + 1) * plan.q, worker_crs),
    ]
    if plan.sharded:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input_tensor))
    else:
        cbs.append(_cb(CB_X_TILES, input_tensor.dtype, sz.T_in, R * plan.WS, worker_crs))
    if out_sharded:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor))
    else:
        cbs.append(_cb(CB_OUTPUT, output_tensor.dtype, sz.T_out, STREAM_DEPTH * plan.C, worker_crs))
    if plan.G_w > 1:
        cbs.append(_cb(CB_PARTIAL_COL, interm, sz.T_i, 2 * plan.q_w, worker_crs))
        cbs.append(_cb(CB_GATHER, interm, sz.T_i, max(1, max_owned) * plan.G_w * plan.q, worker_crs))
    if gamma is not None:
        cbs.append(_cb(CB_GAMMA_TILES, gamma.dtype, sz.T_g, plan.Wp, worker_crs))
        x_gamma_rows = plan.q_w if plan.gamma_per_wave else R
        cbs.append(_cb(CB_X_GAMMA, interm, sz.T_i, x_gamma_rows * plan.WS, worker_crs))
    if plan.masked:
        cbs.append(_cb(CB_SUMSQ_LAST, interm, sz.T_i, plan.q_w, worker_crs))
        cbs.append(_cb(CB_SUMSQ_ACC, interm, sz.T_i, plan.q_w, worker_crs))

    # ---- compile-time args ----
    reader_ct = [
        int(plan.sharded),  # 0
        plan.G_w,  # 1
        plan.WS,  # 2
        plan.num_waves,  # 3
        plan.q_w,  # 4
        plan.owners,  # 5
        plan.q,  # 6
        G,  # 7
        L,  # 8
        sz.T_in,  # 9
        sz.T_i,  # 10
        int(interm == ttnn.float32),  # 11
        int(plan.masked),  # 12
        w_valid_last,  # 13
        _dtype_kind(input_tensor.dtype),  # 14
        SEM_GATHER,  # 15
        SEM_INV,  # 16
        SEM_LANDED_BASE,  # 17
        SEM_OWN_STATS_BASE,  # 18
        READ_LOOKAHEAD_WAVES,  # 19
        seg_stride,  # 20
        Wt,  # 21
        gather_field_bits(plan.G_w),  # 22
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    compute_ct = [
        int(gamma is not None),  # 0
        int(plan.sharded),  # 1
        int(plan.G_w > 1),  # 2
        int(plan.masked),  # 3
        plan.WS,  # 4
        plan.C,  # 5
        plan.nch,  # 6
        plan.num_waves,  # 7
        plan.q_w,  # 8
        plan.owners,  # 9
        plan.q,  # 10
        G,  # 11
        plan.G_w,  # 12
        SCALE_BLOCK_TILES,  # 13
        int(out_sharded),  # 14
        int(plan.gamma_per_wave),  # 15
    ]
    gamma_is_tile = gamma is not None and gamma.layout == ttnn.TILE_LAYOUT
    writer_ct = [
        int(gamma is not None),  # 0
        int(out_sharded),  # 1
        plan.WS,  # 2
        plan.C,  # 3
        plan.nch,  # 4
        plan.num_waves,  # 5
        plan.q_w,  # 6
        Wt,  # 7
        sz.T_out,  # 8
        int(gamma_is_tile),  # 9
        _gamma_es(gamma) if gamma is not None else 0,  # 10
        sz.T_g,  # 11
        plan.Wp,  # 12
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    port_ct = [G, plan.num_waves, int(ttnn.get_tt_fabric_max_payload_size_bytes()), MAX_HEADERS, SEM_OWN_STATS_BASE]

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
        dst_full_sync_en=compute_kernel_config.dst_full_sync_en,
    )
    if fp32_acc:
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        modes[CB_STATS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes

    inv_w_bits = _fp32_bits(1.0 / (G * W_local))
    eps_bits = _fp32_bits(epsilon)
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    groups_list = [groups[g] for g in range(plan.G_h)]
    group_coords = [[pack(virt(grp[j].core)) for j in range(plan.G_w)] for grp in groups_list]
    port_xy = [pack(virt(ports[(d, l)])) for l in range(L) for d in (FWD, BWD)]

    dm_defines = _measurement_defines()
    mesh_desc = ttnn.MeshProgramDescriptor()
    for grp_coords in build_groups(mesh_shape, cluster_axis):
        for p, coord in enumerate(grp_coords):
            info = xinfo[p]
            sends_own = {d: int(len(info["sends"][d]) > 0) for d in (FWD, BWD)}
            n_active = sum(sends_own.values())
            src_dir = [2 if pos == p else (FWD if pos in info["entries"][FWD] else BWD) for pos in range(G)]
            n_in = {d: len(info["entries"][d]) for d in (FWD, BWD)}

            program = ttnn.ProgramDescriptor()
            reader_rt = ttnn.RuntimeArgs()
            writer_rt = ttnn.RuntimeArgs()
            compute_rt = ttnn.RuntimeArgs()
            for m in plan.members:
                c = m.core
                masked_core = int(plan.masked and m.w0 + m.V == Wt)
                rt = [
                    in_addr,
                    m.start_row,
                    m.rows,
                    m.w0,
                    m.V,
                    m.index,
                    p,
                    res.landing_addr,
                    m.group,
                    n_in[FWD],
                    n_in[BWD],
                    int(m.w0 + m.V == Wt),  # holds the W-pad tile (interleaved zeroing)
                    masked_core,
                    n_active,
                ]
                rt += group_coords[m.group]
                rt += port_xy
                rt += [sends_own[FWD], sends_own[BWD]]
                rt += src_dir
                reader_rt[c.x][c.y] = rt
                compute_rt[c.x][c.y] = [m.rows, m.V, m.index, masked_core, inv_w_bits, eps_bits]
                writer_rt[c.x][c.y] = [out_addr, gamma_addr, m.start_row, m.rows, m.w0, m.V]

            # ---- port args ----
            port_rt = ttnn.RuntimeArgs()
            peer_of = {
                FWD: grp_coords[(p + 1) % G] if (ring or p < G - 1) else None,
                BWD: grp_coords[(p - 1) % G] if (ring or p > 0) else None,
            }
            for l in range(L):
                for d in (FWD, BWD):
                    core = ports[(d, l)]
                    sends, ready_out, entries = info["sends"][d], info["ready_out"][d], info["entries"][d]
                    has_conn = len(sends) > 0 or len(ready_out) > 0
                    pc = virt(core)
                    dests = sorted({(dst, h) for _, dst, _, h, _ in sends} | {(t, h) for t, _, h in ready_out})
                    assert len(dests) <= MAX_HEADERS
                    root_list = roots[l]
                    n_notify = len(root_list) if len(entries) > 0 else 0
                    args = [
                        res.landing_addr,
                        seg_stride,
                        res.ready_addr,
                        len(dests),
                        len(sends),
                        len(entries),
                        n_pub[l] if len(sends) > 0 else 0,  # n_roots: publishers into my own slot per wave
                        len(ready_out),
                        len({it[1] for it in sends}),
                        pc.x,
                        pc.y,
                        n_notify,
                        int(has_conn),
                        1,  # notify quantum: per landed entry
                        SEM_LANDED_BASE + 2 * l + d,
                        len(root_list),
                    ]
                    args += res.arrival_addrs[: len(entries)]
                    for dst, h in dests:
                        dn = _node(mesh, grp_coords[dst])
                        args += [int(dn.chip_id), int(dn.mesh_id), h]
                    for q_slot, dst, idx, h, wait in sends:
                        args += [q_slot, dests.index((dst, h)), res.arrival_addrs[idx], wait]
                    for tgt, td, h in ready_out:
                        rc = virt(ports[(td, l)])
                        args += [dests.index((tgt, h)), rc.x, rc.y]
                    args += [seg_bytes[l]] * plan.num_waves
                    args += root_list
                    if has_conn:
                        peer = peer_of[d]
                        me_node, pn = _node(mesh, coord), _node(mesh, peer)
                        link_ids = ttnn.get_forwarding_link_indices(me_node, pn)
                        args += list(ttnn.setup_fabric_connection(me_node, pn, link_ids[l], program, core))
                    port_rt[core.x][core.y] = args

            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
                    core_ranges=worker_crs,
                    compile_time_args=reader_ct,
                    defines=dm_defines,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
                    core_ranges=worker_crs,
                    compile_time_args=writer_ct,
                    defines=dm_defines,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                    core_ranges=worker_crs,
                    compile_time_args=compute_ct,
                    defines=dm_defines,
                    runtime_args=compute_rt,
                    config=compute_cfg,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port.cpp"),
                    core_ranges=port_crs,
                    compile_time_args=port_ct,
                    defines=dm_defines,
                    runtime_args=port_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                ),
            ]
            program.kernels = kernels
            program.cbs = cbs
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=i, core_ranges=all_crs, initial_value=0) for i in range(n_sems)
            ]
            key = "|".join(
                str(v)
                for v in (
                    "distributed_rms_norm",
                    tuple(dm_defines),
                    coord,
                    p,
                    G,
                    ring,
                    L,
                    cluster_axis,
                    plan.regime,
                    plan.S,
                    plan.G_w,
                    plan.G_h,
                    plan.rows_max,
                    plan.num_waves,
                    plan.q_w,
                    plan.owners,
                    tuple(shape),
                    input_tensor.dtype,
                    None if gamma is None else (gamma.dtype, gamma.layout),
                    fp32_acc,
                    compute_kernel_config.math_fidelity,
                    compute_kernel_config.math_approx_mode,
                    mc.memory_layout,
                    None if mc.shard_spec is None else (tuple(mc.shard_spec.shape), str(mc.shard_spec.grid)),
                    output_tensor.memory_config().buffer_type,
                    tuple(sorted(pset)),
                    out_sharded,
                    plan.gamma_per_wave,
                    res.landing_addr,
                )
            )
            program.custom_program_hash = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    return mesh_desc, res
