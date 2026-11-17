# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Port-core placement: each fabric connection's sending core goes next to its link's Ethernet core.

Adapted from ttnn/ttnn/operations/examples/fabric_all_gather (probe_ethernet_cores / _worker_nearest_noc1): a probe
program builds every connection on device (without opening it) and reports the router's NoC coordinates; Blackhole
Ethernet and Tensix coordinates are translated, so both are mapped back to physical columns before matching.
"""

from __future__ import annotations

import os

import ttnn

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

_BH_ETH_TRANSLATED_X0 = 20
_CACHE = {}


def _arch_yaml(mesh_device):
    import yaml

    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    name = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh_device.arch()).split(".")[-1]
    ]
    return yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", name)))


def _chip_lists(mesh_device):
    """{asic unique id: (physical Ethernet core list, noc translation on, translated->physical Tensix column map)}."""
    key = ("lists", id(mesh_device))
    if key in _CACHE:
        return _CACHE[key][1]
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
    _CACHE[key] = (mesh_device, lists)
    return lists


def _noc_grid(mesh_device):
    g = _arch_yaml(mesh_device)["grid"]
    return int(g["x_size"]), int(g["y_size"])


def _chip_maps(mesh_device, coord):
    node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(node.mesh_id), int(node.chip_id)))
    return _chip_lists(mesh_device).get(uid, (None, False, {}))


def _eth_physical(mesh_device, coord, xy):
    x, y = xy
    eth_list, translated, _ = _chip_maps(mesh_device, coord)
    if translated and eth_list is not None and x >= _BH_ETH_TRANSLATED_X0:
        return eth_list[x - _BH_ETH_TRANSLATED_X0]
    return x, y


def probe_ethernet_cores(mesh_device, connections, allowed):
    """connections: {coord: [(peer, link)]} -> {(coord, peer, link): physical (x, y) of that link's Ethernet core}."""
    key = (
        "probe",
        id(mesh_device),
        ttnn.get_fabric_config(),
        tuple(sorted((k, tuple(v)) for k, v in connections.items())),
    )
    if key in _CACHE:
        return _CACHE[key][1]
    import torch

    k_max = max(len(v) for v in connections.values())
    row = next(
        y for y in sorted({y for _, y in allowed}, reverse=True) if sum(1 for _, yy in allowed if yy == y) >= k_max
    )
    cores = [ttnn.CoreCoord(x, row) for x in sorted(x for x, y in allowed if y == row)[:k_max]]
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]), (1, 8), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )
    slots = ttnn.from_torch(
        torch.zeros((k_max, 8), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    slot_addr = int(slots.buffer_address())
    rows, cols = tuple(mesh_device.shape)
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
            result[((r, c), peer, link)] = _eth_physical(mesh_device, (r, c), (int(got[i, 0]), int(got[i, 1])))
    _CACHE[key] = (mesh_device, result)
    return result


def nearest_worker_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Logical worker core (from `allowed`, not taken) with the fewest NoC1 hops (-Y, then -X, wrapping) to the
    Ethernet core at physical eth_xy; ties to the lower row, then the lower column."""
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
            best = (score, (x, y))
    return best[1] if best is not None else None
