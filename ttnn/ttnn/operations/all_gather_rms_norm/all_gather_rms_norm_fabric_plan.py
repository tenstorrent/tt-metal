# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fabric plan of all_gather_rms_norm (TRANSPORT): groups, ring schedules, port-core placement.

Adapted from ttnn/ttnn/operations/examples/fabric_all_gather (plan(), _schedule(), probe_ethernet_cores(),
_worker_nearest_noc1(), _needs_ready()): one port core per (direction, link) per chip, placed next to its link's
Ethernet core (the free worker with the fewest NoC1 hops to it, found by a one-time probe program cached per
mesh); receive-only ports (line ends) on any free core. The port set of every chip is excluded from the compute
grid of EVERY chip (ports_union), so the compute-core list — and with it the mirror layout of cb_x — is identical
on every chip of the mesh.
"""

from __future__ import annotations

import os

import ttnn

OPP = {"fwd": "bwd", "bwd": "fwd"}
DIRS = ("fwd", "bwd")


def build_groups(mesh_shape, cluster_axis):
    """Ordered lists of mesh coords (group-position order), one per group."""
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def schedule(p, G, ring, balance=True):
    """(shards sent toward p+1, shards sent toward p-1), own shard first, in the order they arrive downstream.

    Entries are (position, part): part 0 = the whole shard, 1 / 2 = the first / second half of it (here: the
    row-groups whose (g div L) is even / odd). balance (ring, even G >= 4): the shard opposite each receiver goes
    half forward (part 1), half backward (part 2), so both directions carry G/2 - 1/2 shards (example measured
    1.10-1.28x from ~64 tiles)."""
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


_PROBE_CACHE = {}

_PROBE_SOURCE = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

// Report where this connection's router (Ethernet core) sits: build the connection (without opening it) and write
// the router's NoC coordinates into my L1 slot.
void kernel_main() {
    size_t a = 0;
    const uint32_t slot = get_arg_val<uint32_t>(a++);
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
    p[0] = conn.edm_noc_x;
    p[1] = conn.edm_noc_y;
}
"""


def probe_ethernet_cores(mesh_device, connections, allowed):
    """connections: {coord: [(peer_coord, link_index), ...]} -> {(coord, peer_coord, link_index): (noc_x, noc_y)}.

    One probe program per chip (one-time setup, cached per mesh like the global semaphores)."""
    key = (id(mesh_device), ttnn.get_fabric_config(), tuple(sorted((k, tuple(v)) for k, v in connections.items())))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key][1]
    rows, cols = tuple(mesh_device.shape)
    k_max = max(len(v) for v in connections.values())
    row = next(
        (y for y in sorted({y for _, y in allowed}, reverse=True) if sum(1 for _, yy in allowed if yy == y) >= k_max),
        None,
    )
    if row is None:
        raise RuntimeError(f"all_gather_rms_norm: no row of the core grid has {k_max} cores for the placement probe")
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
_NOC_GRID_CACHE = {}


def _arch_yaml(mesh_device):
    return {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh_device.arch()).split(".")[-1]
    ]


def _eth_physical_lists(mesh_device):
    if id(mesh_device) in _ETH_MAP_CACHE:
        return _ETH_MAP_CACHE[id(mesh_device)][1]
    import yaml

    desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    soc = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", _arch_yaml(mesh_device))))
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


def _noc_grid(mesh_device):
    if id(mesh_device) not in _NOC_GRID_CACHE:
        import yaml

        root = os.environ.get("TT_METAL_HOME", os.getcwd())
        grid = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", _arch_yaml(mesh_device))))["grid"]
        _NOC_GRID_CACHE[id(mesh_device)] = (mesh_device, (int(grid["x_size"]), int(grid["y_size"])))
    return _NOC_GRID_CACHE[id(mesh_device)][1]


def _worker_nearest_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Free logical worker with the fewest NoC1 hops (-Y then -X, wrapping) to the Ethernet core at eth_xy."""
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


def _any_free(taken, allowed, prefer):
    """A free core for a receive-only port: one already used as a port elsewhere in the mesh if possible
    (keeps ports_union small), else the first free core from the top row (next to the Ethernet row)."""
    for xy in prefer:
        if xy not in taken and xy in allowed:
            return ttnn.CoreCoord(*xy)
    for x, y in sorted(allowed, key=lambda c: (c[1], c[0])):
        if (x, y) not in taken:
            return ttnn.CoreCoord(x, y)
    return None


def plan_fabric(mesh_device, *, cluster_axis, ring_requested, num_links):
    """Groups, ring flag, per-chip neighbours, link indices and port placement.

    Returns (chips, ring, ports_union):
      chips[coord] = dict(p, G, next, prev, links={d: [link idx]}, ports={(d, l): CoreCoord})
      ports exist for (d, l) whenever the chip has a peer in direction d (it sends / sends the ready) or an
      upstream in direction d (it receives) — the receive-only port of a line end.
    """
    mesh_shape = tuple(mesh_device.shape)
    groups = build_groups(mesh_shape, cluster_axis)
    G = len(groups[0])
    ring = ring_requested and G > 2  # a 2-chip ring is the line
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            nxt = grp[(p + 1) % G] if (ring or p < G - 1) else None
            prv = grp[(p - 1) % G] if (ring or p > 0) else None
            chips[coord] = dict(p=p, G=G, next=nxt, prev=prv, group=grp, links={})
    connections = {}
    for coord, ch in chips.items():
        conns = []
        for d, peer in (("fwd", ch["next"]), ("bwd", ch["prev"])):
            if peer is None:
                continue
            links = list(ttnn.get_forwarding_link_indices(node(coord), node(peer)))
            if len(links) < num_links:
                raise ValueError(
                    f"all_gather_rms_norm: {coord} -> {peer} has {len(links)} usable link(s), {num_links} requested "
                    f"(fabric={ttnn.get_fabric_config()})"
                )
            ch["links"][d] = links[:num_links]
            conns += [(peer, links[l]) for l in range(num_links)]
        connections[coord] = conns
    grid = mesh_device.compute_with_storage_grid_size()
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    placement_auto = not os.environ.get("TT_METAL_EMULE_MODE")
    eth = probe_ethernet_cores(mesh_device, connections, allowed) if placement_auto else None
    union_pref = []
    for coord, ch in chips.items():
        taken = set()
        ports = {}
        for d, peer in (("fwd", ch["next"]), ("bwd", ch["prev"])):
            if peer is None:
                continue
            for l in range(num_links):
                core = None
                if eth is not None:
                    exy = _eth_noc_coord(mesh_device, coord, eth[(coord, peer, ch["links"][d][l])])
                    core = _worker_nearest_noc1(mesh_device, coord, exy, taken, allowed)
                if core is None:
                    core = _any_free(taken, allowed, union_pref)
                ports[(d, l)] = core
                taken.add((core.x, core.y))
                union_pref.append((core.x, core.y))
        ch["ports"] = ports
        ch["taken"] = taken
    # receive-only ports (no peer in d, but an upstream sends in d): line ends
    for coord, ch in chips.items():
        for d in DIRS:
            upstream = ch["prev"] if d == "fwd" else ch["next"]
            peer = ch["next"] if d == "fwd" else ch["prev"]
            if peer is None and upstream is not None:
                for l in range(num_links):
                    core = _any_free(ch["taken"], allowed, union_pref)
                    if core is None:
                        raise RuntimeError("all_gather_rms_norm: the core grid has too few cores for the ports")
                    ch["ports"][(d, l)] = core
                    ch["taken"].add((core.x, core.y))
                    union_pref.append((core.x, core.y))
    ports_union = sorted(
        {(c.x, c.y) for ch in chips.values() for c in ch["ports"].values()}, key=lambda c: (c[1], c[0])
    )
    return chips, ring, ports_union


def resolve_num_links(mesh_device, cluster_axis, ring_requested):
    """Usable links per hop = the minimum over the group's hops (num_links=None)."""
    mesh_shape = tuple(mesh_device.shape)
    groups = build_groups(mesh_shape, cluster_axis)
    G = len(groups[0])
    ring = ring_requested and G > 2
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    best = None
    for grp in groups:
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring:
            hops.append((grp[-1], grp[0]))
        for a, b in hops:
            for x, y in ((a, b), (b, a)):
                n = len(ttnn.get_forwarding_link_indices(node(x), node(y)))
                best = n if best is None else min(best, n)
    return best or 0
