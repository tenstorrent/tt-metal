# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for ring_joint_scaled_dot_product_attention (regime R1 `qstationary_dram_kv`).

Per device (one ProgramDescriptor per mesh coordinate, one MeshProgramDescriptor, one generic_op dispatch):

  Port cores (TRANSPORT, adapted from examples/fabric_all_gather): one per (direction, link). The port reader
  (NCRISC, NoC0) reads this device's own K/V head slices (entries = (batch, kv head), head-major), then the slices it
  relays from the DRAM landing buffers, each relay only once the arrival counter says the upstream slice landed; the
  port sender (BRISC, NoC1) sends each chunk (a run of <= payload/page tiles consecutive in one DRAM bank) into the
  downstream device's landing slot, the last chunk of every (entry, source) item as a fused write + increment of the
  downstream port's arrival counter. Per-call ready fence; at the end the sender re-arms its counter once every
  upstream item landed and every compute reader of this device stopped polling it (consumers_done).

  Compute cores (attention, the rest of the grid): attention units (b, q head, q block of Sq tiles) round-robin
  over the cores (head-major, so waves advance in lockstep with the head-major transport). Each unit walks its key
  segments local -> joint -> remote by hop distance in blocks of Sk tiles (online softmax). The reader gates each
  remote segment on the arrival counters of the ports carrying it (OVERLAP: the local and joint blocks hide the
  first landing).

Every block knob (q_block_tiles, k_block_tiles, buffer depths, port batch) is defined once here and flows to the
kernels as compile-time args; CB page counts derive from them.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- block knobs (single source of truth) ---------------------------------------------------------------------
BATCHES_PER_UNIT = 1  # batch axis: nothing is shared across batches
# q_head axis (design regime R3): q heads stacked into one unit's row space, sharing ONE key-block stream (K/V DRAM
# reads / g). None = the whole share group hq / max(hk, hv) (GQA/MQA: hq/hk; MHA and MLA: 1, since MLA's V heads are
# per q head); an int must divide that group. The single source of g: every unit decode derives from it.
Q_HEADS_PER_UNIT = None
Q_BLOCK_DEPTH = 1  # cb_q_block depth (Q retained for the unit's whole key walk)
KV_BLOCK_DEPTH = 2  # cb_k_block / cb_v_block depth: reader one key block ahead of compute
OUT_BLOCK_DEPTH = 1  # cb_out_tiles depth: the writer drains a unit during the next unit's walk
PORT_BATCH_CHUNKS = 4  # chunks per port batch; cb_port_chunks holds 2 batches (fabric_all_gather README)
L1_CB_BUDGET = 1300 * 1024  # per-core CB budget for choose_block_extents (design: ~1.3 MB on Blackhole)
# Precision policy (compute_kernel_config is a FLOOR): with fp32 DEST accumulation requested, the compute kernel runs
# at least this fidelity. Measured (golden test_regression::test_large_logits, logit std ~36, bf16, fp32 DEST): the
# Q.K^T operand truncation of HiFi2 gives rel-RMS 0.0206 (gate 0.02); HiFi3 0.0045, HiFi4 0.0045. Not applied
# when fp32_dest_acc_en is off (the DiT production / perf-focus config keeps the requested HiFi2).
FP32_DEST_MIN_FIDELITY = ttnn.MathFidelity.HiFi3
MASK_TILES = 1  # main key-tail mask tile (joint tail mask comes with non-aligned joint)
CAUSAL_MASK_TILES = 2  # causal (R4): [diagonal triangle, all -inf]; the key-tail mask is not needed (see writer)

# ---- CB indices (compute cores) ----
CB_Q_BLOCK = 0
CB_K_BLOCK = 1
CB_V_BLOCK = 2
CB_KEY_MASK = 3
CB_MAX_SCALER = 4
CB_SUM_SCALER = 5
CB_SCORES = 6
CB_MAX_BLK = 7
CB_MAX_ACC = 8
CB_SUM_BLK = 9
CB_SUM_ACC = 10
CB_ROW_FACTOR = 11
CB_OUT_BLK = 12
CB_OUT_ACC = 13
CB_OUT_TILES = 16
# Credit-only CB (no payload): the streamed bf16-DEST block posts one token per finished P row (pack thread) so P.V /
# rowsum (unpack) never read a row before its in-place exp landed. 4 pages x 32 B = 128 B, outside compute_cb_plan.
CB_P_TOKEN = 14
P_TOKEN_DEPTH = 4
# ---- CB indices (port cores) ----
CB_PORT_CHUNKS = 0

NOC0 = ttnn.NOC.RISCV_0_default
NOC1 = ttnn.NOC.RISCV_1_default
_OPP = {"fwd": "bwd", "bwd": "fwd"}


def _pow2_floor(x):
    return 1 << (max(1, int(x)).bit_length() - 1)


def _pow2_ceil(x):
    x = max(1, int(x))
    return 1 << (x - 1).bit_length()


def q_heads_per_unit(hq, hk, hv):
    """g: q heads stacked per unit (R3). Heads in one unit must share both their K and their V head."""
    share = hq // max(hk, hv)
    g = share if Q_HEADS_PER_UNIT is None else int(Q_HEADS_PER_UNIT)
    if g < 1 or share % g:
        raise ValueError(f"ring_joint_sdpa: q_heads_per_unit={g} must divide hq / max(hk, hv) = {share}")
    return g


def compute_cb_plan(Sq, Sk, *, DHt, DVt, q_dtype, kv_dtype, acc_dtype, mask_tiles=MASK_TILES):
    """The compute-core CB inventory as (index, pages, dtype) — the single source of every compute CB's size.

    choose_block_extents() sums it for the L1 fit and the program descriptor allocates exactly it, so a change to a
    CB (or a buffer-depth knob) lands in one place (ledger: l1_ledger.md).

    16-bit DEST (acc_dtype != float32) runs the streamed block, which corrects l_acc / O_acc in place (single
    accumulators), so the per-block scratch cb_sum_blk / cb_out_blk exist only for the fp32-DEST body."""
    single_acc = acc_dtype != ttnn.float32
    plan = [
        (CB_Q_BLOCK, Q_BLOCK_DEPTH * Sq * DHt, q_dtype),
        (CB_K_BLOCK, KV_BLOCK_DEPTH * Sk * DHt, kv_dtype),
        (CB_V_BLOCK, KV_BLOCK_DEPTH * Sk * DVt, kv_dtype),
        (CB_KEY_MASK, mask_tiles, ttnn.bfloat16),
        (CB_MAX_SCALER, 1, ttnn.bfloat16),
        (CB_SUM_SCALER, 1, ttnn.bfloat16),
        (CB_SCORES, Sq * Sk, acc_dtype),
        (CB_MAX_BLK, Sq, acc_dtype),
        (CB_MAX_ACC, Sq, acc_dtype),
        (CB_SUM_BLK, Sq, acc_dtype),
        (CB_SUM_ACC, Sq, acc_dtype),
        (CB_ROW_FACTOR, Sq, acc_dtype),
        (CB_OUT_BLK, Sq * DVt, acc_dtype),
        (CB_OUT_ACC, Sq * DVt, acc_dtype),
        (CB_OUT_TILES, OUT_BLOCK_DEPTH * Sq * DVt, ttnn.bfloat16),
    ]
    if single_acc:
        plan = [e for e in plan if e[0] not in (CB_SUM_BLK, CB_OUT_BLK)]
    return plan


def choose_block_extents(
    *,
    tensor_q_tiles,
    joint_q_tiles,
    tensor_kv_tiles,
    b,
    hq,
    C,
    DHt,
    DVt,
    q_dtype,
    kv_dtype,
    acc_dtype,
    mask_tiles=MASK_TILES,
):
    """The single source of (q_block_tiles, k_block_tiles) — op_design.md `choose_block_extents()`. With stacked
    heads (R3) tensor_q_tiles / joint_q_tiles are a unit row space's (g heads) and hq counts head groups."""

    def footprint(Sq, Sk):
        plan = compute_cb_plan(
            Sq, Sk, DHt=DHt, DVt=DVt, q_dtype=q_dtype, kv_dtype=kv_dtype, acc_dtype=acc_dtype, mask_tiles=mask_tiles
        )
        return sum(pages * ttnn.tile_size(dt) for _, pages, dt in plan)

    total_q_tiles = b * hq * (tensor_q_tiles + joint_q_tiles)
    sq_occ = _pow2_floor(max(1, total_q_tiles // max(1, C)))
    sk_cap = _pow2_ceil(tensor_kv_tiles)
    Sq = min(sq_occ, _pow2_ceil(tensor_q_tiles))
    while Sq >= 1:
        Sk = sk_cap
        while Sk >= 1 and footprint(Sq, Sk) > L1_CB_BUDGET:
            Sk //= 2
        if Sk >= 1 and Sk >= min(4, sk_cap):
            return Sq, Sk
        Sq //= 2
    Sk = sk_cap
    while Sk >= 1 and footprint(1, Sk) > L1_CB_BUDGET:
        Sk //= 2
    if Sk < 1:
        raise ValueError("ring_joint_sdpa: no (q_block, k_block) fits the L1 CB budget")
    return 1, Sk


# Tail-piece cost model (plan_work_split), in units of ONE q tile-row's compute per key block. Measured on the focus
# shape (BH QB2, Sq 16, Sk 8, dh 128, bf16 K/V, bf16 DEST): a unit's key block costs F + r c with F ~ 3.5 us fixed
# and c ~ 7.9 us per q tile-row; p concurrent tail pieces each reading a 128 KB K/V block share ~335 GB/s of DRAM,
# 0.39 us per piece per block. Tail time ~ n_key_blocks * max(F + r c, p d): r=1 (64 pieces) 3.2 ms, DRAM-bound;
# r=2 (32 pieces) 2.5 ms, r=4 (16) 4.5 ms, compute-bound — the model ranks them the same way.
TAIL_FIXED_ROWS = 0.45  # F / c
TAIL_READ_PER_PIECE = 0.05  # d / c for a 2 KB K/V tile (scaled by the K/V tile size)


def plan_work_split(Sq, *, n_q_t, Lq_t, BH, C, kv_tile_bytes=2048, tail_rows=None):
    """Balanced tail split. Units are Sq-row q blocks of one (batch, q head), head-major ([main blocks], [joint
    blocks] per head) as in ring_joint_sdpa_common.hpp. The first W = U // C waves stay whole units round-robin
    (core c owns units c, c + C, ..., c + (W-1) C); the remainder units (the last U - W*C) are re-blocked into
    `tail pieces` of r q tile-rows (r a power of two dividing Sq, picked by the cost model below among the r whose
    piece count fits C cores) and
    piece c goes to core c AFTER its full units. A piece's compute runs on r rows (cb credits r-sized; r | Sq keeps
    every block contiguous in its CB, and the piece is the core's last unit so nothing follows it).

    Returns (W, pieces, r): pieces[c] = (bh, row0, rows, joint) for c < len(pieces) <= C."""
    assert Sq & (Sq - 1) == 0, "Sq must be a power of two"
    nqb_main = -(-n_q_t // Sq)
    nqb_joint = -(-Lq_t // Sq)
    Uq = nqb_main + nqb_joint
    U = BH * Uq
    W, rem = divmod(U, C)
    if rem == 0:
        return W, [], Sq
    tail = []
    for u in range(W * C, U):
        bh, qb = divmod(u, Uq)
        if qb < nqb_main:
            row0 = qb * Sq
            tail.append((bh, row0, min(Sq, n_q_t - row0), 0))
        else:
            row0 = (qb - nqb_main) * Sq
            tail.append((bh, row0, min(Sq, Lq_t - row0), 1))

    def split(r):
        out = []
        for bh, row0, rows, joint in tail:
            for o in range(0, rows, r):
                out.append((bh, row0 + o, min(r, rows - o), joint))
        return out

    if tail_rows is not None:
        r = max(1, min(Sq, _pow2_floor(int(tail_rows))))
        while r < Sq and len(split(r)) > C:
            r *= 2
    else:
        d = TAIL_READ_PER_PIECE * kv_tile_bytes / 2048
        cands = []
        r_ = 1
        while r_ <= Sq:
            n = len(split(r_))
            if n <= C:
                cands.append((max(TAIL_FIXED_ROWS + r_, n * d), r_))
            r_ *= 2
        r = min(cands)[1]
    pieces = split(r)
    assert len(pieces) <= C
    return W, pieces, r


# ----------------------------------------------------------------------------------------------------------------
# Transport plan (TRANSPORT) — adapted from examples/fabric_all_gather (groups, schedule, links, port placement)
# ----------------------------------------------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _schedule(p, G, ring):
    """(sources sent toward p+1, sources sent toward p-1), own first, in the order they arrive downstream."""
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [(p - i) % G for i in range(kf)], ([(p + i) % G for i in range(kb)] if kb > 0 else [])
    fwd = [p - i for i in range(p + 1)] if p < G - 1 else []
    bwd = [p + i for i in range(G - p)] if p > 0 else []
    return fwd, bwd


def _slot(p, s, G):
    """Landing slot of source s on device p."""
    return (s - p - 1) % G


_PROBE_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

void kernel_main() {
    size_t a = 0;
    const uint32_t slot = get_arg_val<uint32_t>(a++);
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    p[0] = conn.edm_noc_x;
    p[1] = conn.edm_noc_y;
}
"""

_PROBE_CACHE = {}


def _probe_ethernet_cores(mesh_device, connections, allowed):
    """{coord: [(peer, link)]} -> {(coord, peer, link): (noc_x, noc_y)} of the router (one probe program per mesh)."""
    key = (id(mesh_device), ttnn.get_fabric_config(), tuple(sorted((k, tuple(v)) for k, v in connections.items())))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key][1]
    rows, cols = tuple(mesh_device.shape)
    k_max = max([len(v) for v in connections.values()] + [1])
    row = next(
        (y for y in sorted({y for _, y in allowed}, reverse=True) if sum(1 for _, yy in allowed if yy == y) >= k_max),
        None,
    )
    if row is None:
        raise ValueError(f"ring_joint_sdpa: no row of the core grid has {k_max} cores for the placement probe")
    cores = [ttnn.CoreCoord(x, row) for x in sorted(x for x, y in allowed if y == row)[:k_max]]
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_set, (1, 8), ttnn.ShardOrientation.ROW_MAJOR),
    )
    import torch

    slots = ttnn.from_torch(
        torch.zeros((k_max, 8), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    slot_addr = int(slots.buffer_address())
    mesh_desc = ttnn.MeshProgramDescriptor()
    for (r, c), conns in connections.items():
        me = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
        program = ttnn.ProgramDescriptor()
        rt = ttnn.RuntimeArgs()
        used = []
        for i, (peer, link) in enumerate(conns):
            peer_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*peer))
            rt[cores[i].x][cores[i].y] = [slot_addr] + list(
                ttnn.setup_fabric_connection(me, peer_node, link, program, cores[i])
            )
            used.append(cores[i])
        if used:
            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_PROBE_SOURCE,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(u, u) for u in used]),
                    compile_time_args=[],
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0),
                )
            ]
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    ttnn.generic_op([slots, slots], mesh_desc)
    ttnn.synchronize_device(mesh_device)
    per_dev = [ttnn.to_torch(t).reshape(k_max, 8) for t in ttnn.get_device_tensors(slots)]
    result = {}
    for (r, c), conns in connections.items():
        got = per_dev[r * cols + c]
        for i, (peer, link) in enumerate(conns):
            result[((r, c), peer, link)] = (int(got[i, 0]), int(got[i, 1]))
    _PROBE_CACHE[key] = (mesh_device, result)
    return result


_ETH_MAP_CACHE = {}
_BH_ETH_TRANSLATED_X0 = 20


def _soc_yaml(mesh_device):
    import yaml

    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh_device.arch()).split(".")[-1]
    ]
    return yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))


def _eth_physical_lists(mesh_device):
    if id(mesh_device) in _ETH_MAP_CACHE:
        return _ETH_MAP_CACHE[id(mesh_device)][1]
    import yaml

    desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
    soc = _soc_yaml(mesh_device)
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
    _ETH_MAP_CACHE[id(mesh_device)] = (mesh_device, lists)
    return lists


def _chip_maps(mesh_device, coord):
    node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(node.mesh_id), int(node.chip_id)))
    return _eth_physical_lists(mesh_device).get(uid, (None, False, {}))


def _eth_noc_coord(mesh_device, coord, xy):
    x, y = xy
    eth_list, translated, _ = _chip_maps(mesh_device, coord)
    if translated and eth_list is not None and x >= _BH_ETH_TRANSLATED_X0:
        return eth_list[x - _BH_ETH_TRANSLATED_X0]
    return x, y


_NOC_GRID_CACHE = {}


def _noc_grid(mesh_device):
    if id(mesh_device) not in _NOC_GRID_CACHE:
        grid = _soc_yaml(mesh_device)["grid"]
        _NOC_GRID_CACHE[id(mesh_device)] = (mesh_device, (int(grid["x_size"]), int(grid["y_size"])))
    return _NOC_GRID_CACHE[id(mesh_device)][1]


def _worker_nearest_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Free worker core with the fewest NoC1 hops (-Y then -X, wrapping) to the Ethernet core at eth_xy."""
    t2p = _chip_maps(mesh_device, coord)[2]
    gx, gy = _noc_grid(mesh_device)
    ex, ey = eth_xy
    best = None
    for x, y in sorted(allowed):
        if (x, y) in taken:
            continue
        v = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        px = t2p.get(v.x, v.x)
        score = ((v.y - ey) % gy + (px - ex) % gx, v.y, px)
        if best is None or score < best[0]:
            best = (score, ttnn.CoreCoord(x, y))
    return best[1] if best is not None else None


def _worker_any(taken, allowed):
    for y in sorted({y for _, y in allowed}, reverse=True):
        for x in sorted(x for x, yy in allowed if yy == y):
            if (x, y) not in taken:
                return ttnn.CoreCoord(x, y)
    return None


def _allowed_cores(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return {(x, y) for x in range(grid.x) for y in range(grid.y)}


_PLAN_CACHE = {}


def plan(mesh_device, *, cluster_axis, topology, num_links):
    """Groups, per-direction source schedules, links per hop and port placement for every chip (cached)."""
    key = (id(mesh_device), ttnn.get_fabric_config(), cluster_axis, int(topology == Topology.Ring), num_links)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key][1]
    mesh_shape = tuple(mesh_device.shape)
    groups = build_groups(mesh_shape, cluster_axis)
    G = len(groups[0])
    if G < 2:
        raise ValueError(f"ring_joint_sdpa: cluster_axis={cluster_axis} has {G} device(s); a group needs >= 2")
    ring = topology == Topology.Ring and G > 2  # a 2-device ring runs the line schedule
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            nxt = grp[(p + 1) % G] if (ring or p < G - 1) else None
            prv = grp[(p - 1) % G] if (ring or p > 0) else None
            fwd, bwd = _schedule(p, G, ring)
            chips[coord] = dict(p=p, G=G, group=grp, next=nxt, prev=prv, fwd=fwd, bwd=bwd)

    def needs_ready(coord, d):
        ch = chips[coord]
        peer = ch["next"] if d == "fwd" else ch["prev"]
        return peer is not None and bool(chips[peer][_OPP[d]])

    # links per used hop; num_links None = the minimum over the group's hops
    hop_links = {}
    for coord, ch in chips.items():
        for d, peer in (("fwd", ch["next"]), ("bwd", ch["prev"])):
            if peer is None or not (ch[d] or needs_ready(coord, d)):
                continue
            links = list(ttnn.get_forwarding_link_indices(node(coord), node(peer)))
            if not links:
                raise ValueError(
                    f"ring_joint_sdpa: no usable fabric link {coord} -> {peer} "
                    f"(topology={topology}, fabric={ttnn.get_fabric_config()})"
                )
            hop_links[(coord, d)] = links
    usable = min(len(v) for v in hop_links.values())
    if num_links is None:
        num_links = usable
    if num_links < 1 or num_links > usable:
        raise ValueError(f"ring_joint_sdpa: num_links={num_links} outside [1, {usable}] usable on every hop")
    allowed = _allowed_cores(mesh_device)
    connections = {coord: [] for coord in chips}
    for (coord, d), links in hop_links.items():
        ch = chips[coord]
        peer = ch["next"] if d == "fwd" else ch["prev"]
        ch[f"{d}_links"] = links[:num_links]
        connections[coord] += [(peer, links[l]) for l in range(num_links)]
    auto = not (os.environ.get("TT_METAL_EMULE_MODE") or os.environ.get("TT_METAL_SIMULATOR"))
    eth = _probe_ethernet_cores(mesh_device, connections, allowed) if auto else None
    for coord, ch in chips.items():
        taken = set()
        ports = {}
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            for l in range(num_links):
                core = None
                if eth is not None and ch.get(f"{d}_links"):
                    exy = _eth_noc_coord(mesh_device, coord, eth[(coord, peer, ch[f"{d}_links"][l])])
                    core = _worker_nearest_noc1(mesh_device, coord, exy, taken, allowed)
                ports[(d, l)] = core
                if core is not None:
                    taken.add((core.x, core.y))
        for k, core in ports.items():  # receive-only ports (line ends) or simple placement: any free core
            if core is None:
                ports[k] = _worker_any(taken, allowed)
                if ports[k] is None:
                    raise ValueError("ring_joint_sdpa: the core grid has too few cores for the port cores")
                taken.add((ports[k].x, ports[k].y))
        ch["ports"] = ports
        ch["compute"] = [ttnn.CoreCoord(x, y) for (x, y) in sorted(allowed - taken, key=lambda t: (t[1], t[0]))]
    result = (chips, ring, G, num_links)
    _PLAN_CACHE[key] = (mesh_device, result)
    return result


def _chunks_per_part(part_pages, first_bank, stride, num_banks, run_pages):
    """Chunks for_each_chunk visits for one part of `part_pages` pages on a port's bank set (host replica)."""
    total = 0
    for b in range(first_bank, num_banks, stride):
        per_bank = (part_pages - b + num_banks - 1) // num_banks if b < part_pages else 0
        total += (per_bank + run_pages - 1) // run_pages
    return total


# ----------------------------------------------------------------------------------------------------------------
# Persistent resources (cached per structural key; never re-created per call)
# ----------------------------------------------------------------------------------------------------------------

_SEM_CACHE = {}
_LANDING_CACHE = {}


def _semaphores(mesh_device):
    """(arrival, ready, consumers_done, poll_scratch) global semaphores over the whole worker grid, created once per
    mesh. poll_scratch is never a counter: a compute reader NoC-reads a port's arrival count into its own copy of it.
    It must not be one of the counters — port placement changes between calls (topology / num_links), so a core
    that polled as a compute core in one call can be a port core in the next, and a polled value left in a counter
    word would read as already-landed items there."""
    grid = mesh_device.compute_with_storage_grid_size()
    key = (id(mesh_device), grid.x, grid.y)
    if key not in _SEM_CACHE:
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(4))
        # every chip's counters must be zero before any chip's first increment
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, sems)
    return _SEM_CACHE[key][1]


def _landing_buffers(mesh_device, G, k, v):
    b, hk, n_kv, dh = tuple(k.padded_shape)
    _, hv, _, dv = tuple(v.padded_shape)
    key = (id(mesh_device), G, b, hk, hv, n_kv, dh, dv, str(k.dtype), str(v.dtype))
    if key not in _LANDING_CACHE:
        lk = ttnn.allocate_tensor_on_device(
            ttnn.Shape([(G - 1) * b, hk, n_kv, dh]), k.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        lv = ttnn.allocate_tensor_on_device(
            ttnn.Shape([(G - 1) * b, hv, n_kv, dv]), v.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        _LANDING_CACHE[key] = (mesh_device, (lk, lv))
    return _LANDING_CACHE[key][1]


def _check_tensor(t, what):
    if t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"ring_joint_sdpa: {what} must be TILE layout")
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"ring_joint_sdpa: {what} must be DRAM interleaved")
    if len(t.shape) != 4:
        raise ValueError(f"ring_joint_sdpa: {what} must be rank 4")


def _bf16_bits(x):
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF  # round to nearest even


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


# ----------------------------------------------------------------------------------------------------------------
# Entry: build + dispatch
# ----------------------------------------------------------------------------------------------------------------


def run_ring_joint_sdpa(
    q,
    k,
    v,
    jq,
    jk,
    jv,
    *,
    logical_n,
    logical_l,
    cluster_axis,
    topology,
    num_links,
    scale,
    compute_kernel_config,
    memory_config,
    is_causal=False,
    is_balanced=False,
):
    mesh_device = q.device()
    joints = (jq, jk, jv)
    if any(t is None for t in joints) and not all(t is None for t in joints):
        raise ValueError("ring_joint_sdpa: joint_tensor_q/k/v are all given or all None")
    has_joint = jq is not None
    for t, what in ((q, "Q"), (k, "K"), (v, "V")) + (
        ((jq, "joint Q"), (jk, "joint K"), (jv, "joint V")) if has_joint else ()
    ):
        _check_tensor(t, what)
    b, hq, n_q, dh = tuple(q.shape)
    kb_, hk, n_kv, dhk = tuple(k.shape)
    vb_, hv, n_v, dv = tuple(v.shape)
    if kb_ != b or vb_ != b or dhk != dh or n_v != n_kv:
        raise ValueError(f"ring_joint_sdpa: mismatched Q/K/V shapes {q.shape} {k.shape} {v.shape}")
    if hq % hk or hq % hv:
        raise ValueError("ring_joint_sdpa: hq must be a multiple of hk and hv")
    if k.dtype != v.dtype:
        raise ValueError("ring_joint_sdpa: K and V must share a dtype")
    L = 0
    if has_joint:
        jb, jhq, L, jdh = tuple(jq.shape)
        if tuple(jk.shape) != (b, hk, L, dh) or tuple(jv.shape) != (b, hv, L, dv) or (jb, jhq, jdh) != (b, hq, dh):
            raise ValueError("ring_joint_sdpa: joint tensor shapes do not match Q/K/V")
        if jq.dtype != q.dtype or jk.dtype != k.dtype or jv.dtype != v.dtype:
            raise ValueError("ring_joint_sdpa: joint tensor dtypes must match Q/K/V")
    if logical_l != 0:
        raise ValueError("ring_joint_sdpa: sharded joint (logical_l > 0) is not supported")
    if is_balanced and not is_causal:
        raise ValueError("ring_joint_sdpa: is_balanced is defined for causal attention only")
    if is_causal and (has_joint or n_q != n_kv):
        raise ValueError("ring_joint_sdpa: causal attention is self-attention without joint tokens")
    if is_balanced and n_kv % 64:
        raise ValueError(f"ring_joint_sdpa: balanced needs two tile-aligned halves per device; n={n_kv}")

    chips, ring, G, num_links = plan(mesh_device, cluster_axis=cluster_axis, topology=topology, num_links=num_links)
    if not (1 <= logical_n <= G * n_kv):
        raise ValueError(f"ring_joint_sdpa: logical_n={logical_n} outside [1, {G * n_kv}]")
    if scale is None:
        scale = dh**-0.5

    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    math_fidelity = compute_kernel_config.math_fidelity
    if fp32_acc and math_fidelity.value < FP32_DEST_MIN_FIDELITY.value:
        math_fidelity = FP32_DEST_MIN_FIDELITY
    acc_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16
    DHt, DVt = dh // 32, dv // 32
    n_q_t, n_kv_t = n_q // 32, n_kv // 32
    Lq_t = (L + 31) // 32
    C = len(next(iter(chips.values()))["compute"])
    qh = q_heads_per_unit(hq, hk, hv)
    NQ, LQ, BG = qh * n_q_t, qh * Lq_t, b * hq // qh  # unit row space: g heads stacked (contiguous tile rows)
    mask_tiles = CAUSAL_MASK_TILES if is_causal else MASK_TILES
    Sq, Sk = choose_block_extents(
        tensor_q_tiles=NQ,
        joint_q_tiles=LQ,
        tensor_kv_tiles=n_kv_t,
        b=b,
        hq=hq // qh,
        C=C,
        DHt=DHt,
        DVt=DVt,
        q_dtype=q.dtype,
        kv_dtype=k.dtype,
        acc_dtype=acc_dtype,
        mask_tiles=mask_tiles,
    )

    cb_plan = compute_cb_plan(
        Sq, Sk, DHt=DHt, DVt=DVt, q_dtype=q.dtype, kv_dtype=k.dtype, acc_dtype=acc_dtype, mask_tiles=mask_tiles
    )
    W_full, tail_pieces, tail_r = plan_work_split(
        Sq, n_q_t=NQ, Lq_t=LQ, BH=BG, C=C, kv_tile_bytes=ttnn.tile_size(k.dtype)
    )
    out_mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([b, hq, n_q, dv]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, out_mem
    )
    jout = (
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([b, hq, L, dv]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, out_mem
        )
        if has_joint
        else None
    )
    landk, landv = _landing_buffers(mesh_device, G, k, v)
    sem_arrival, sem_ready, sem_cdone, sem_poll = _semaphores(mesh_device)
    arrival_addr, ready_addr, cdone_addr, poll_addr = (
        int(ttnn.get_global_semaphore_address(s)) for s in (sem_arrival, sem_ready, sem_cdone, sem_poll)
    )

    page_bytes = int(k.buffer_aligned_page_size())
    assert page_bytes == int(v.buffer_aligned_page_size())
    g = mesh_device.dram_grid_size()
    num_banks = g.x * g.y
    if num_links > num_banks:
        raise ValueError(f"ring_joint_sdpa: {num_links} links need at least as many DRAM banks ({num_banks})")
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)
    group = PORT_BATCH_CHUNKS
    kpe, vpe = n_kv_t * DHt, n_kv_t * DVt  # pages of one K / V head slice
    Hmax = max(hk, hv)
    E = b * Hmax  # transport entries (b, j): K head j if j < hk, V head j if j < hv

    def valid_rows(s):
        """Rows of device s at a global position < logical_n (a prefix: positions grow along a device)."""
        if not is_balanced:
            return max(0, min(n_kv, logical_n - s * n_kv))
        h = n_kv // 2
        first = max(0, min(h, logical_n - s * h))
        return first + (max(0, min(h, logical_n - (2 * G - 1 - s) * h)) if first == h else 0)

    mask_cols = logical_n % 32

    addr = lambda t: int(t.buffer_address()) if t is not None else 0
    ta = (
        lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        if t is not None
        else list(ttnn.TensorAccessorArgs().get_compile_time_args())
    )
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    scale_bits = _bf16_bits(scale)

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        program = ttnn.ProgramDescriptor()
        ports = ch["ports"]
        port_keys = [(d, l) for d in ("fwd", "bwd") for l in range(num_links)]
        port_cores = [ports[pk] for pk in port_keys]
        compute = ch["compute"]
        part_chunks = {
            l: (
                _chunks_per_part(kpe, l, num_links, num_banks, run_pages),
                _chunks_per_part(vpe, l, num_links, num_banks, run_pages),
            )
            for l in range(num_links)
        }
        item_chunks = {l: ck + cv for l, (ck, cv) in part_chunks.items()}
        for l, (ck, cv) in part_chunks.items():  # every item of a used port must carry >= 1 chunk (its increment)
            if ck + cv and ((hk < hv and cv == 0) or (hv < hk and ck == 0)):
                raise ValueError(f"ring_joint_sdpa: a K/V head slice is too small to split over {num_links} links")

        # ---------------- port cores (TRANSPORT) ----------------
        preader_rt, psender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (d, l), core in ports.items():
            peer = ch["next"] if d == "fwd" else ch["prev"]
            sends = ch[d] if peer is not None else []
            up = ch["prev"] if d == "fwd" else ch["next"]
            up_sends = chips[up][d] if up is not None else []
            ck, cv = part_chunks[l]
            ic = ck + cv
            total = b * (hk * ck + hv * cv) * len(sends)
            expect_in = E * len(up_sends) if ic > 0 else 0
            preader_rt[core.x][core.y] = [
                addr(k),
                addr(v),
                addr(landk),
                addr(landv),
                kpe,
                vpe,
                l,
                num_links,
                arrival_addr,
                ck,
                cv,
                total,
                E,
                hk,
                hv,
                len(sends),
                len(up_sends),
            ] + [_slot(p, s, G) for s in sends]
            send_ready = peer is not None and bool(chips[peer][_OPP[d]])
            rc = virt(chips[peer]["ports"][(_OPP[d], l)]) if send_ready else None
            has_conn = bool(ch.get(f"{d}_links"))
            args = [
                addr(landk),
                addr(landv),
                kpe,
                vpe,
                l,
                num_links,
                arrival_addr,
                expect_in,
                total,
                ck,
                cv,
                ready_addr,
                int(send_ready),
                rc.x if rc else 0,
                rc.y if rc else 0,
            ]
            if has_conn:
                pc = virt(chips[peer]["ports"][(d, l)])
                pn = node(peer)
                q_pos = chips[peer]["p"]
                args += [pc.x, pc.y, int(pn.mesh_id), int(pn.chip_id), cdone_addr, C, E, hk, hv, len(sends), 1]
                args += [_slot(q_pos, s, G) for s in sends]
                args += list(ttnn.setup_fabric_connection(node(coord), pn, ch[f"{d}_links"][l], program, core))
            else:
                args += [0, 0, 0, 0, cdone_addr, C, E, hk, hv, 0, 0]
            psender_rt[core.x][core.y] = args

        # ---------------- compute-core key segments (local -> joint -> remote by hop distance) ----------------
        port_index = {pk: i for i, pk in enumerate(port_keys)}
        segments = []  # (kind, valid_tiles, masked, slot, S_in, rank, port_mask, src)

        def add_main(kind, s, slot=0, S_in=0, rank=0, port_mask=0):
            vr = valid_rows(s)
            if vr == 0:
                return
            # causal: a key past logical_n is only visible to padded queries (key <= query), so no tail mask
            masked = int(vr % 32 != 0 and not is_causal)
            segments.append((kind, (vr + 31) // 32, masked, slot, S_in, rank, port_mask, s))

        add_main(0, p)
        if has_joint:
            segments.append((1, Lq_t, 0, 0, 0, 0, 0, 0))
        arr = {}
        for d in ("fwd", "bwd"):
            up = ch["prev"] if d == "fwd" else ch["next"]
            up_sends = chips[up][d] if up is not None else []
            arr[d] = up_sends
        for dist in range(1, G):
            for d in ("fwd", "bwd"):
                if dist - 1 < len(arr[d]):
                    s = arr[d][dist - 1]
                    mask = 0
                    for l in range(num_links):
                        if item_chunks[l] > 0:
                            mask |= 1 << port_index[(d, l)]
                    add_main(2, s, _slot(p, s, G), len(arr[d]), dist - 1, mask)
        seg_flat = [x for sgm in segments for x in sgm]
        seg_compute = [x for sgm in segments for x in (sgm[1], sgm[2], sgm[7])]
        pxy = [virt(pc) for pc in port_cores]

        # ---------------- compute-core CBs ----------------
        ccrs = _crs(compute)

        def cb(idx, pages, dtype):
            page = ttnn.tile_size(dtype)
            return ttnn.CBDescriptor(
                total_size=pages * page,
                core_ranges=ccrs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
            )

        program.cbs = [cb(*e) for e in cb_plan] + [
            ttnn.CBDescriptor(
                total_size=P_TOKEN_DEPTH * 32,
                core_ranges=ccrs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_P_TOKEN, data_format=ttnn.bfloat16, page_size=32)
                ],
            ),
            ttnn.CBDescriptor(
                total_size=2 * group * run_pages * page_bytes,
                core_ranges=_crs(port_cores),
                format_descriptors=[ttnn.CBFormatDescriptor(CB_PORT_CHUNKS, k.dtype, page_bytes)],
            ),
        ]

        # ---------------- compute-core kernels ----------------
        unit_ct = [Sq, Sk, DHt, DVt, NQ, LQ, n_kv_t, BG, C, hq, hk, hv, qh, n_q_t, Lq_t]
        unit_ct += [int(is_causal), int(is_balanced), G]
        reader_ct = [CB_Q_BLOCK, CB_K_BLOCK, CB_V_BLOCK] + unit_ct + [len(port_cores)]
        for t in (q, k, v, jq, jk, jv, landk, landv):
            reader_ct += ta(t)
        writer_ct = [CB_OUT_TILES, CB_KEY_MASK, CB_MAX_SCALER, CB_SUM_SCALER] + unit_ct + [mask_cols]
        writer_ct += ta(out) + ta(jout)
        compute_ct = (
            [
                CB_Q_BLOCK,
                CB_K_BLOCK,
                CB_V_BLOCK,
                CB_KEY_MASK,
                CB_MAX_SCALER,
                CB_SUM_SCALER,
                CB_SCORES,
                CB_MAX_BLK,
                CB_MAX_ACC,
                CB_SUM_BLK,
                CB_SUM_ACC,
                CB_ROW_FACTOR,
                CB_OUT_BLK,
                CB_OUT_ACC,
                CB_OUT_TILES,
            ]
            + unit_ct
            + [scale_bits, CB_P_TOKEN]
        )

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()

        def tail_args(ci):  # has_tail, bh, row0, rows, joint
            return [1] + list(tail_pieces[ci]) if ci < len(tail_pieces) else [0, 0, 0, 0, 0]

        for ci, core in enumerate(compute):
            reader_rt[core.x][core.y] = (
                [
                    addr(q),
                    addr(k),
                    addr(v),
                    addr(jq),
                    addr(jk),
                    addr(jv),
                    addr(landk),
                    addr(landv),
                    ci,
                    arrival_addr,
                    cdone_addr,
                    poll_addr,
                ]
                + [c.x for c in pxy]
                + [c.y for c in pxy]
                + [len(segments)]
                + seg_flat
                + [W_full]
                + tail_args(ci)
                + [p]
            )
            writer_rt[core.x][core.y] = [addr(out), addr(jout), ci, W_full] + tail_args(ci) + [tail_r]
            compute_rt[core.x][core.y] = [ci, len(segments)] + seg_compute + [W_full] + tail_args(ci) + [tail_r, p]

        prefix = "ring_joint_scaled_dot_product_attention"
        port_crs = _crs(port_cores)
        port_ct = [CB_PORT_CHUNKS, page_bytes, run_pages, num_banks, group]
        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{prefix}_port_reader.cpp"),
                core_ranges=port_crs,
                compile_time_args=port_ct + ta(k) + ta(v) + ta(landk) + ta(landv),
                runtime_args=preader_rt,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{prefix}_port_sender.cpp"),
                core_ranges=port_crs,
                compile_time_args=port_ct + ta(landk) + ta(landv),
                runtime_args=psender_rt,
                config=dm("RISCV_0", NOC1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{prefix}_reader.cpp"),
                core_ranges=ccrs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{prefix}_writer.cpp"),
                core_ranges=ccrs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"{prefix}_compute.cpp"),
                core_ranges=ccrs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=math_fidelity,
                    fp32_dest_acc_en=fp32_acc,
                    math_approx_mode=False,
                ),
            ),
        ]
        # generic_op's cache ignores the per-core RT-arg layout: key on everything structural
        hkey = "|".join(
            str(x)
            for x in (
                prefix,
                coord,
                G,
                ring,
                num_links,
                cluster_axis,
                b,
                hq,
                hk,
                hv,
                qh,
                is_causal,
                is_balanced,
                n_q,
                n_kv,
                dh,
                dv,
                L,
                logical_n,
                Sq,
                Sk,
                W_full,
                tail_r,
                len(tail_pieces),
                q.dtype,
                k.dtype,
                fp32_acc,
                math_fidelity,
                scale_bits,
                run_pages,
                tuple((c.x, c.y) for c in port_cores),
                len(compute),
                tuple(segments),
                has_joint,
                out_mem,  # the writer's out / joint_out TensorAccessorArgs are compile-time
            )
        )
        program.custom_program_hash = int.from_bytes(hashlib.sha1(hkey.encode()).digest()[:8], "little") >> 1
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program

    io = [q, k, v] + ([jq, jk, jv, jout] if has_joint else []) + [landk, landv, out]
    ttnn.generic_op(io, mesh_desc)
    return out, jout
