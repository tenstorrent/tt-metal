# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm_kimi: fused all-gather (hidden dim scattered over a ring) + RMSNorm, one ttnn.generic_op.

Every device of a group (a mesh row for cluster_axis 1) holds x[p] of shape (1, 1, M, W_local); every device returns
rms_norm(concat_p x[p], gamma) of shape (1, 1, M, W), W = G * W_local.

Design (per chip, identical core layout on every chip of the mesh):
  * Row cores: one per (row block b, group slice p). Core (b, p) owns the tile-rows of block b and the Wt = W_local / 32
    tile columns of slice p. Its persistent L1 landing buffer (two halves, one per call parity) receives slice p's block
    straight from the fabric (p remote) or from DRAM (p = this chip). It computes S = sum_j x_j^2 (fp32), sends S to
    the G - 1 other cores of its block (one write per core: S tile + tag), row-reduces the G S tiles in fp32 DEST in
    slice order (bit-identical on every core and chip), takes rsqrt(. / W + eps) on column 0, and writes its normalized
    Wt output tiles.
  * Port cores: one per (ring direction, link), next to the link's Ethernet core (probed). Each sends one-hop fabric
    writes from its staging buffer, in a host-fixed order (_schedule): this chip's own slice for the row blocks of its
    link (block b -> link b % L), and peers' packets that have hops left (a ring relay: the receiving row core pushes
    them to its port). On an even ring of G, toward p + 1 the first half of each tile-row goes G/2 hops and the second
    G/2 - 1, toward p - 1 the other way round: every chip receives each remote slice once and both directions carry
    (G - 1) / 2 slices per link. Relays are one-hop sends, so the ring may turn (the QuietBox 1x4 snake does).
  * Packets carry no semaphore increment: each payload ends in a 16 B tag (call counter + 1, a per-state random salt)
    that receivers poll. No fence between calls: landing buffers alternate by call parity (see port_sender.cpp).
  * Local row cores read their slice in send order (a token chain, PUSH_AHEAD rows ahead) and push packets to the
    ports; ports fetch their first PORT_FETCH packets from DRAM themselves; remote row cores read gamma only after
    their first packet lands (so ~700 tiny gamma reads don't queue in DRAM ahead of the first packets).

Persistent state (per mesh, shape, axis, links): one global semaphore (the call counter, on every op core) and the
landing buffers (2 x RowLayout slots per row core, ~56 KB of L1 per row core, zero-filled at creation).
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.ccl import Topology

from . import placement

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_LAND, CB_GAMMA, CB_SCALER, CB_SQ, CB_PART, CB_GATHER, CB_INV, CB_XG, CB_OUT, CB_STAGE = range(10)
SEM_GATHER = 0
SEM_TOKEN = 1
PUSH_AHEAD = 3  # local row cores of a link that may push ahead of the token chain
TILE_BF16 = 2048
TILE_FP32 = 4096


# ---------------------------------------------------------------------------------------------------------------------
# Registry model
# ---------------------------------------------------------------------------------------------------------------------


def default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.HiFi4
    config.fp32_dest_acc_en = True
    config.math_approx_mode = False
    return config


def _tag_alignment(inputs, axes):
    shape = inputs[0]
    if shape[-1] % 32 != 0:
        return "w_non_aligned"
    if shape[-2] % 32 != 0:
        return "h_non_aligned"
    return "tile_aligned"


def _tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "alignment": _tag_alignment,
    "rank": _tag_rank,
}

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "fp32_dest_acc_en": [True],
    "layout": [ttnn.TILE_LAYOUT],
    "alignment": ["tile_aligned"],
    "rank": [4],
    "gamma_dtype": [ttnn.bfloat16],
    "gamma_layout": [ttnn.ROW_MAJOR_LAYOUT],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED],
    "cluster_axis": [1],
    "topology": [Topology.Ring],
    "num_links": [2],
}

EXCLUSIONS = []


def validate(
    input_tensor, *, gamma, cluster_axis, epsilon=1e-6, topology=Topology.Ring, num_links=2, compute_kernel_config=None
):
    cfg = compute_kernel_config or default_compute_kernel_config()
    axes = {
        "dtype": input_tensor.dtype,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "layout": input_tensor.layout,
        "gamma_dtype": gamma.dtype,
        "gamma_layout": gamma.layout,
        "memory_layout": input_tensor.memory_config().memory_layout,
        "cluster_axis": cluster_axis,
        "topology": topology,
        "num_links": num_links,
    }
    shape = tuple(input_tensor.shape)
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger((shape,), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"all_gather_rms_norm_kimi: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"all_gather_rms_norm_kimi: unsupported combination {exc}")


# ---------------------------------------------------------------------------------------------------------------------
# Plan: groups, ring directions, port placement, row-core layout, packet schedules (cached per mesh + shape)
# ---------------------------------------------------------------------------------------------------------------------

_PLANS = {}


def _node(mesh_device, coord):
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def _slice_routes(G, Wt):
    """Own slice of chip p: [(direction, col0, ncols, hops)], direction +1 = toward p + 1, -1 = toward p - 1.
    Each remote chip receives each column exactly once."""
    if G % 2 == 0:
        h = G // 2
        a = (Wt + 1) // 2  # == _row_layout
        routes = [(+1, 0, a, h), (+1, a, Wt - a, h - 1), (-1, a, Wt - a, h), (-1, 0, a, h - 1)]
    else:
        h = (G - 1) // 2
        routes = [(+1, 0, Wt, h), (-1, 0, Wt, h)]
    return [r for r in routes if r[2] > 0 and r[3] > 0]


def _plan(mesh_device, M, W_local, cluster_axis, num_links):
    key = (id(mesh_device), ttnn.get_fabric_config(), tuple(mesh_device.shape), M, W_local, cluster_axis, num_links)
    if key in _PLANS:
        return _PLANS[key][1]
    rows_m, cols_m = tuple(mesh_device.shape)
    G = (rows_m, cols_m)[cluster_axis]
    if G < 2:
        raise ValueError(f"all_gather_rms_norm_kimi: cluster_axis {cluster_axis} has size {G}; nothing to gather")
    if cluster_axis == 1:
        groups = [[(r, c) for c in range(cols_m)] for r in range(rows_m)]
    else:
        groups = [[(r, c) for r in range(rows_m)] for c in range(cols_m)]
    L = num_links
    Wt, R = W_local // 32, M // 32

    # ring neighbours, links and fabric directions
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            nxt, prv = grp[(p + 1) % G], grp[(p - 1) % G]
            ch = dict(p=p, group=grp, next=nxt, prev=prv)
            for name, peer in (("next", nxt), ("prev", prv)):
                links = list(ttnn.get_forwarding_link_indices(_node(mesh_device, coord), _node(mesh_device, peer)))
                if len(links) < L:
                    raise ValueError(
                        f"all_gather_rms_norm_kimi: {coord} -> {peer} has {len(links)} usable link(s), {L} requested "
                        f"(a Ring needs a direct link on every hop, including the wrap; fabric "
                        f"{ttnn.get_fabric_config()})"
                    )
                d = ttnn.get_eth_forwarding_direction(_node(mesh_device, coord), _node(mesh_device, peer))
                if d is None:
                    raise ValueError(f"all_gather_rms_norm_kimi: no fabric route {coord} -> {peer}")
                ch[name + "_links"] = links[:L]
                ch[name + "_dir"] = int(d)
            chips[coord] = ch
    routes = _slice_routes(G, Wt)

    # port cores
    grid = mesh_device.compute_with_storage_grid_size()
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    connections = {
        coord: [(ch[n], ch[n + "_links"][l]) for n in ("next", "prev") for l in range(L)] for coord, ch in chips.items()
    }
    eth = placement.probe_ethernet_cores(mesh_device, connections, allowed)
    used = set()
    for coord, ch in chips.items():
        taken = set()
        ports = {}
        for n in ("next", "prev"):
            for l in range(L):
                exy = eth[(coord, ch[n], ch[n + "_links"][l])]
                core = placement.nearest_worker_noc1(mesh_device, coord, exy, taken, allowed)
                if core is None:
                    raise ValueError("all_gather_rms_norm_kimi: no free core for a port")
                taken.add(core)
                ports[(n, l)] = core
        ch["ports"] = ports
        used |= taken

    # row cores: (row block b, slice p), the same logical cores on every chip
    free = sorted(allowed - used, key=lambda xy: (xy[1], xy[0]))
    max_blocks = len(free) // G
    rows_per = next((k for k in range(1, R + 1) if R % k == 0 and R // k <= max_blocks), None)
    if rows_per is None:
        raise ValueError(f"all_gather_rms_norm_kimi: {R} tile-rows do not fit the {len(free)}-core grid")
    nblk = R // rows_per
    row_cores = {(b, p): free[b * G + p] for b in range(nblk) for p in range(G)}

    plan = dict(
        G=G,
        L=L,
        Wt=Wt,
        R=R,
        rows_per=rows_per,
        nblk=nblk,
        chips=chips,
        row_cores=row_cores,
        routes=routes,
        all_cores=sorted(used | set(row_cores.values())),
        port_cores=sorted(used),
    )
    plan["max_pkt_tiles"] = max(1, min(255, ttnn.get_tt_fabric_max_payload_size_bytes() // TILE_BF16))
    plan["sched"] = _schedule(plan, plan["max_pkt_tiles"])
    plan["half_a"], slots, _ = _row_layout(G, Wt, plan["max_pkt_tiles"])
    plan["blk"] = slots * rows_per
    _PLANS[key] = (mesh_device, plan)
    return plan


def _row_layout(G, Wt, P):
    """(a, slots, slot_of): the landing layout of one tile-row (kernels/agrn_common.hpp RowLayout)."""
    a = (Wt + 1) // 2 if G % 2 == 0 else Wt
    ka = -(-a // P)
    npk = ka + -(-(Wt - a) // P)
    slot_of = lambda j: j + j // P if j < a else j + ka + (j - a) // P
    return a, Wt + npk, slot_of


RELAY_LAG = 3  # a relay of tile-row t is sent after this port's own packets of tile-row t + RELAY_LAG
FAR_LEAD = 1  # an own relay-bound half of tile-row t is sent with the own direct halves of tile-row t - FAR_LEAD


def _schedule(plan, max_pkt_tiles):
    """Every port's send list for the whole group, simulated on the host (the kernels only follow the lists).

    A packet carries `n` tiles of slice `s`, tile-row block `b`, landing tile slot `off`; it lands in the downstream
    chip's row core (b, s); `rem` = hops still to go after that (rem > 0: that row core pushes it to its own port of the
    same direction and link, which sends it on with rem - 1). Order per port: own packets of its t-th tile-row, then the
    forwarded ones RELAY_LAG tile-rows later (they need a hop to arrive first)."""
    G, L, Wt, rows_per, nblk = plan["G"], plan["L"], plan["Wt"], plan["rows_per"], plan["nblk"]
    _, slots, slot_of = _row_layout(G, Wt, max_pkt_tiles)
    ports = [(q, sgn, l) for q in range(G) for sgn in (+1, -1) for l in range(L)]
    link_rows = {l: [b * rows_per + i for b in range(l, nblk, L) for i in range(rows_per)] for l in range(L)}
    own = {}
    for q, sgn, l in ports:
        items = []
        for t, row in enumerate(link_rows[l]):
            b, i = divmod(row, rows_per)
            for _, c0, nc, hops in sorted((r for r in plan["routes"] if r[0] == sgn), key=lambda r: -r[3]):
                for t0 in range(0, nc, max_pkt_tiles):
                    n = min(max_pkt_tiles, nc - t0)
                    # relay-bound halves go FAR_LEAD tile-rows early: the last packets of a call are then one-hop ones
                    key = (t - FAR_LEAD if hops > 1 else t, 0, 0)
                    items.append(
                        dict(
                            key=key,
                            t=t,
                            relay=False,
                            row=row,
                            c=c0 + t0,
                            n=n,
                            b=b,
                            s=q,
                            off=i * slots + slot_of(c0 + t0),
                            rem=hops - 1,
                        )
                    )
        own[(q, sgn, l)] = items
    sends = {k: list(v) for k, v in own.items()}
    relay_in = {}
    for _ in range(G):
        relay_in = {}
        for q, sgn, l in ports:
            up = sends[((q - sgn) % G, sgn, l)]
            relay_in[(q, sgn, l)] = [it for it in up if it["rem"] > 0]
        for q, sgn, l in ports:
            fwd = []
            for it in relay_in[(q, sgn, l)]:
                t = it["t"]
                dist = it["key"][2] + 1
                # sent RELAY_LAG tile-rows after upstream sent it (it went FAR_LEAD rows early if it was an own half)
                sent_at = it["key"][0]
                fwd.append(
                    dict(
                        key=(sent_at + RELAY_LAG, 1, dist),
                        t=t,
                        relay=True,
                        n=it["n"],
                        b=it["b"],
                        s=it["s"],
                        off=it["off"],
                        rem=it["rem"] - 1,
                    )
                )
            sends[(q, sgn, l)] = sorted(own[(q, sgn, l)] + fwd, key=lambda it: it["key"])
    return dict(sends=sends)


# ---------------------------------------------------------------------------------------------------------------------
# Persistent state: call counter + arrival counters (global semaphores), landing buffers (L1, both parities)
# ---------------------------------------------------------------------------------------------------------------------

_STATE = {}


def _state(mesh_device, plan, key):
    if key in _STATE:
        return _STATE[key][1]
    import torch

    ctr = ttnn.create_global_semaphore(mesh_device, _crs(plan["all_cores"]), 0)  # the call counter
    row_list = [plan["row_cores"][(b, p)] for b in range(plan["nblk"]) for p in range(plan["G"])]
    tiles = 2 * plan["blk"]
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_crs(row_list), (tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Zero-filled: a packet is recognized by its tag (call counter + 1, counted from 0 per state), so leftovers of an
    # earlier process at the same L1 address could otherwise carry a matching tag.
    land = ttnn.from_torch(
        torch.zeros((1, 1, len(row_list) * tiles * 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.synchronize_device(mesh_device)  # the counter is zero on every chip before any chip's first call
    st = dict(ctr=ctr, land=land, salt=int.from_bytes(os.urandom(4), "little") | 1)
    _STATE[key] = (mesh_device, st)
    return st


# ---------------------------------------------------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------------------------------------------------

PKT_GROUP = 4  # packets between sender header flushes
PORT_FETCH = 4  # own packets each port reads from DRAM itself (its first sends)


def _crs(cores):
    """Cores as a CoreRangeSet merged into rectangles: single-core ranges multiply the dispatch work (binaries, CB
    configs and semaphores are written per range) and made fused-CCL ops dispatch-bound."""
    s = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    return s.merge(s)


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _build(mesh_device, plan, st, x, gamma, out, epsilon, cfg):
    G, Wt, rows_per, nblk = plan["G"], plan["Wt"], plan["rows_per"], plan["nblk"]
    sched = plan["sched"]
    blk = plan["blk"]
    W = G * Wt * 32
    P = plan["max_pkt_tiles"]
    half_a = plan["half_a"]
    _, slots, slot_of = _row_layout(G, Wt, P)
    ka = -(-half_a // P)
    npk_row = ka + -(-(Wt - half_a) // P)
    pk_first = lambda k: k * P if k < ka else half_a + (k - ka) * P
    n_max = max(it["n"] for v in sched["sends"].values() for it in v)
    stage_slot_bytes = n_max * TILE_BF16 + 64  # payload + tag, 64 B aligned
    n_items_max = max(len(v) for v in sched["sends"].values())
    ctr_addr = int(ttnn.get_global_semaphore_address(st["ctr"]))
    land_addr = int(st["land"].buffer_address())
    in_addr, g_addr, out_addr = int(x.buffer_address()), int(gamma.buffer_address()), int(out.buffer_address())
    in_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    g_ct = list(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    dest_tiles = 4 if cfg.fp32_dest_acc_en else 8
    virt = lambda xy: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
    vxy = lambda xy: (lambda v: (v.x << 16) | v.y)(virt(xy))

    row_list = [plan["row_cores"][(b, p)] for b in range(nblk) for p in range(G)]
    row_set = _crs(row_list)
    all_set = _crs(plan["all_cores"])

    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = cfg.math_fidelity
    ccfg.fp32_dest_acc_en = bool(cfg.fp32_dest_acc_en)
    ccfg.math_approx_mode = bool(cfg.math_approx_mode)

    def cb(idx, ntiles, dtype, page, cores):
        return ttnn.CBDescriptor(
            total_size=ntiles * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(idx, dtype, page)],
        )

    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    NOC0, NOC1 = ttnn.NOC.RISCV_0_default, ttnn.NOC.RISCV_1_default
    zones = [("AGRN_ZONES", "1")] if os.environ.get("AGRN_ZONES") else []
    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in plan["chips"].items():
        q = ch["p"]
        program = ttnn.ProgramDescriptor()
        port_set = _crs(list(ch["ports"].values()))
        program.cbs = [
            # first, on every core of the op: the same address everywhere, so row cores can write a port's slots
            cb(CB_STAGE, n_items_max, ttnn.bfloat16, stage_slot_bytes, all_set),
            ttnn.cb_descriptor_from_sharded_tensor(CB_LAND, st["land"]),
            cb(CB_GAMMA, Wt, ttnn.bfloat16, TILE_BF16, row_set),
            cb(CB_SCALER, 1, ttnn.bfloat16, TILE_BF16, row_set),
            # S tiles travel with a tag right after the tile: page = tile + 64 B
            cb(CB_PART, 2, ttnn.float32, TILE_FP32 + 64, row_set),
            cb(CB_GATHER, rows_per * G, ttnn.float32, TILE_FP32 + 64, row_set),
            cb(CB_INV, 2, ttnn.float32, TILE_FP32, row_set),
            cb(CB_XG, Wt, ttnn.float32, TILE_FP32, row_set),
            cb(CB_OUT, Wt, ttnn.bfloat16, TILE_BF16, row_set),
        ]
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=row_set, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_TOKEN, core_ranges=row_set, initial_value=0),
        ]
        # push-token chain over this chip's local row cores (slice q), per link, in send order: block t of a link
        # passes the token to block t + PUSH_AHEAD
        token = {}
        for l in range(plan["L"]):
            blocks = list(range(l, nblk, plan["L"]))
            for t, b in enumerate(blocks):
                nxt = blocks[t + PUSH_AHEAD] if t + PUSH_AHEAD < len(blocks) else None
                token[b] = (int(t >= PUSH_AHEAD), vxy(plan["row_cores"][(nxt, q)]) if nxt is not None else 0)

        # where each packet held by a row core of this chip goes next: (b, s, landing slot) -> [(port core, item)]
        # (the first PORT_FETCH own packets of each port are read from DRAM by the port itself)
        fwd = {}
        self_fetch = {}
        for (n, l), core in ch["ports"].items():
            sgn = +1 if n == "next" else -1
            own_seen = 0
            for idx, it in enumerate(sched["sends"][(q, sgn, l)]):
                if not it["relay"] and own_seen < PORT_FETCH:
                    own_seen += 1
                    self_fetch.setdefault((n, l), []).append((idx, it["row"] * Wt + it["c"], it["n"]))
                    continue
                fwd.setdefault((it["b"], it["s"], it["off"]), []).append((vxy(core), idx))

        # ---- row cores ----
        rd_rt, cm_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for b in range(nblk):
            peers = [vxy(plan["row_cores"][(b, p)]) for p in range(G)]
            for p in range(G):
                cx, cy = plan["row_cores"][(b, p)]
                rd = [in_addr, g_addr, ctr_addr, int(p == q), b * rows_per * Wt, Wt, p * Wt, st["salt"], SEM_TOKEN]
                rd += list(token[b]) if p == q else [0, 0]
                for i in range(rows_per):
                    for k in range(npk_row):
                        dl = fwd.get((b, p, i * slots + slot_of(pk_first(k))), [])
                        rd += [len(dl)] + [w for xy, idx in dl for w in (xy, idx)]
                rd_rt[cx][cy] = rd
                cm_rt[cx][cy] = [ctr_addr]
                wr_rt[cx][cy] = [
                    out_addr,
                    ctr_addr,
                    SEM_GATHER,
                    p,
                    b * rows_per * G * Wt + p * Wt,
                    G * Wt,
                    st["salt"],
                ] + peers

        # ---- port cores ----
        ps_rt, pr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (n, l), core in ch["ports"].items():
            sgn = +1 if n == "next" else -1
            items = sched["sends"][(q, sgn, l)]
            peer = ch[n]
            pn = _node(mesh_device, peer)
            args = [ctr_addr, land_addr, int(pn.chip_id), int(pn.mesh_id), st["salt"], len(items)]
            for it in items:
                args += [vxy(plan["row_cores"][(it["b"], it["s"])]), (it["n"] << 20) | it["off"]]
            args += list(
                ttnn.setup_fabric_connection(
                    _node(mesh_device, coord), pn, ch[n + "_links"][l], program, ttnn.CoreCoord(*core)
                )
            )
            ps_rt[core[0]][core[1]] = args
            sf = self_fetch.get((n, l), [])
            pr_rt[core[0]][core[1]] = [in_addr, ctr_addr, st["salt"], len(sf)] + [w for e in sf for w in e]

        program.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "row_reader.cpp"),
                core_ranges=row_set,
                compile_time_args=[TILE_BF16, Wt, rows_per, half_a, P, stage_slot_bytes] + in_ct + g_ct,
                runtime_args=rd_rt,
                defines=zones,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "row_writer.cpp"),
                core_ranges=row_set,
                compile_time_args=[TILE_BF16, TILE_FP32, Wt, rows_per, G, dest_tiles] + out_ct,
                runtime_args=wr_rt,
                defines=zones,
                config=dm("RISCV_0", NOC1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "row_compute.cpp"),
                core_ranges=row_set,
                compile_time_args=[Wt, rows_per, G, _f32_bits(1.0 / W), _f32_bits(epsilon), dest_tiles, half_a, P],
                runtime_args=cm_rt,
                defines=zones,
                config=ccfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "port_reader.cpp"),
                core_ranges=port_set,
                compile_time_args=[TILE_BF16, stage_slot_bytes] + in_ct,
                runtime_args=pr_rt,
                defines=zones,
                config=dm("RISCV_1", NOC0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "port_sender.cpp"),
                core_ranges=port_set,
                compile_time_args=[TILE_BF16, stage_slot_bytes, PKT_GROUP, blk],
                runtime_args=ps_rt,
                defines=zones,
                config=dm("RISCV_0", NOC1),
            ),
        ]
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc


# ---------------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------------


def _check_inputs(input_tensor, gamma, cluster_axis):
    shape = tuple(input_tensor.shape)
    mesh_device = input_tensor.device()
    G = tuple(mesh_device.shape)[cluster_axis]
    W = G * shape[-1]
    want = (1, 1, math.ceil(W / 32), 32)
    if tuple(gamma.shape) != want:
        raise ValueError(
            f"all_gather_rms_norm_kimi: gamma must be ROW_MAJOR sticks of shape {want}, got " f"{tuple(gamma.shape)}"
        )
    if any(d != 1 for d in shape[:-2]):
        raise ValueError(f"all_gather_rms_norm_kimi: input must be (1, 1, M, W_local), got {shape}")
    mc = input_tensor.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM:
        raise ValueError("all_gather_rms_norm_kimi: input must be DRAM interleaved")
    gmc = gamma.memory_config()
    if gmc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("all_gather_rms_norm_kimi: gamma must be interleaved")


def all_gather_rms_norm_kimi(
    input_tensor,
    *,
    gamma,
    cluster_axis,
    epsilon=1e-6,
    topology=Topology.Ring,
    num_links=2,
    compute_kernel_config=None,
):
    validate(
        input_tensor,
        gamma=gamma,
        cluster_axis=cluster_axis,
        epsilon=epsilon,
        topology=topology,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    cfg = compute_kernel_config or default_compute_kernel_config()
    _check_inputs(input_tensor, gamma, cluster_axis)
    mesh_device = input_tensor.device()
    shape = tuple(input_tensor.shape)
    M, W_local = shape[-2], shape[-1]
    plan = _plan(mesh_device, M, W_local, cluster_axis, num_links)
    key = (id(mesh_device), ttnn.get_fabric_config(), tuple(mesh_device.shape), M, W_local, cluster_axis, num_links)
    st = _state(mesh_device, plan, key)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, M, plan["G"] * W_local]),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    desc = _build(mesh_device, plan, st, input_tensor, gamma, out, epsilon, cfg)
    return ttnn.generic_op([input_tensor, gamma, st["land"], out], desc)
