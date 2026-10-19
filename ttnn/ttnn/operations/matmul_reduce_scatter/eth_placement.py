# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ethernet-aware transport placement: the logical worker core with the fewest NoC1 hops to a link's Ethernet core.

From ttnn/ttnn/operations/examples/fabric_all_gather (mstaletovic/matmul-reduce-scatter-perf), which is not on main;
host-only, from the cluster and SoC descriptors.
"""

import ttnn

_ETH_MAP_CACHE = {}
# Blackhole, with NoC translation on:
#   Ethernet: core i (i-th entry of the chip's non-harvested Ethernet list) has translated coordinate (20 + i, 25).
#   Tensix:   columns are compacted - the k-th live column gets the k-th column's x; harvested columns move to the end.
#             So a live column right of a harvested one has translated x != physical x.
_BH_ETH_TRANSLATED_X0 = 20


def _eth_physical_lists(mesh_device):
    """{asic unique id: [physical (x, y) of Ethernet core i]} from the cluster descriptor + the arch SoC descriptor."""
    if id(mesh_device) in _ETH_MAP_CACHE:
        return _ETH_MAP_CACHE[id(mesh_device)][1]
    import os
    import yaml

    desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh_device.arch()).split(".")[-1]
    ]
    arch_eth = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))["eth"]
    arch_eth = [tuple(int(v) for v in e.split("-")) for e in arch_eth]
    soc = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))
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


_NOC_GRID_CACHE = {}


def _noc_grid(mesh_device):
    """(x_size, y_size) of the chip's NoC grid, from the arch SoC descriptor."""
    if id(mesh_device) not in _NOC_GRID_CACHE:
        import os
        import yaml

        root = os.environ.get("TT_METAL_HOME", os.getcwd())
        arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
            str(mesh_device.arch()).split(".")[-1]
        ]
        grid = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))["grid"]
        _NOC_GRID_CACHE[id(mesh_device)] = (mesh_device, (int(grid["x_size"]), int(grid["y_size"])))
    return _NOC_GRID_CACHE[id(mesh_device)][1]


def _chip_maps(mesh_device, coord):
    node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(node.mesh_id), int(node.chip_id)))
    return _eth_physical_lists(mesh_device).get(uid, (None, False, {}))


def _worker_nearest_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Logical worker core (from `allowed`, not yet taken) with the fewest NoC1 hops to the Ethernet core at physical
    eth_xy; ties to the lower row, then the lower column. NoC1 (the sender's NoC) routes -Y, then -X, wrapping around
    the grid, so the hop count is directional: a core one column left of a harvested column would wrap the whole row.
    Two links whose Ethernet cores sit over one live and one harvested column then get workers in two columns (as
    fabric_all_gather's C++ op places them), instead of both in the live column, where their writes to the routers
    would share the column's NoC1 segment into the Ethernet row."""
    t2p = _chip_maps(mesh_device, coord)[2]
    gx, gy = _noc_grid(mesh_device)
    ex, ey = eth_xy
    best = None
    for x, y in sorted(allowed):
        lc = ttnn.CoreCoord(x, y)
        if (x, y) in taken:
            continue
        v = mesh_device.worker_core_from_logical_core(lc)
        px = t2p.get(v.x, v.x)
        score = ((v.y - ey) % gy + (px - ex) % gx, v.y, px)
        if best is None or score < best[0]:
            best = (score, lc)
    return best[1] if best is not None else None
