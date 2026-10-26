# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ethernet-core maps for port placement: a fabric sender works best right below its router's Ethernet core."""

import os

import yaml

import ttnn

_ARCH_YAML = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}
_SOC_CACHE = {}


def _soc(mesh_device):
    """({asic unique id: (physical (x, y) of each live Ethernet core, translated->physical tensix column map)}, NoC
    grid (x_size, y_size)), read once per mesh from the cluster descriptor and the arch SoC descriptor. No chips when
    either is unavailable (an unknown arch, no SoC descriptor where TT_METAL_HOME points): the ports are then placed
    without the Ethernet map."""
    hit = _SOC_CACHE.get(id(mesh_device))
    if hit is not None and hit[0] is mesh_device:
        return hit[1]
    try:
        with open(ttnn.cluster.serialize_cluster_descriptor()) as f:
            desc = yaml.safe_load(f)
        root = os.environ.get("TT_METAL_HOME", os.getcwd())
        arch_yaml = _ARCH_YAML[str(mesh_device.arch()).split(".")[-1]]
        with open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)) as f:
            soc = yaml.safe_load(f)
    except (KeyError, OSError, yaml.YAMLError):
        out = ({}, (0, 0))
        _SOC_CACHE[id(mesh_device)] = (mesh_device, out)
        return out
    arch_eth = [tuple(int(v) for v in e.split("-")) for e in soc["eth"]]
    tensix_cols = sorted({int(w.split("-")[0]) for w in soc["functional_workers"]})
    chips = {}
    for chip, uid in desc.get("chip_unique_ids", {}).items():
        h = desc.get("harvesting", {}).get(chip, {})
        eth_mask = int(h.get("eth_harvesting_mask", 0))
        tensix_mask = int(h.get("harvest_mask", 0))
        live = [x for i, x in enumerate(tensix_cols) if not (tensix_mask >> i) & 1]
        t2p = {tensix_cols[k]: px for k, px in enumerate(live)} if h.get("noc_translation", False) else {}
        chips[int(uid)] = ([e for i, e in enumerate(arch_eth) if not (eth_mask >> i) & 1], t2p)
    out = (chips, (int(soc["grid"]["x_size"]), int(soc["grid"]["y_size"])))
    _SOC_CACHE[id(mesh_device)] = (mesh_device, out)
    return out


def _chip(mesh_device, coord):
    node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_device, int(node.mesh_id), int(node.chip_id)))
    return _soc(mesh_device)[0].get(uid, (None, {}))


def chip_eth_cores(mesh_device, coord):
    """Physical (x, y) of the chip's live Ethernet cores, indexed by Ethernet channel, or None when unknown."""
    return _chip(mesh_device, coord)[0]


def worker_nearest_noc1(mesh_device, coord, eth_xy, taken, allowed):
    """Logical worker core (from `allowed`, not yet taken) with the fewest NoC1 hops to the Ethernet core at physical
    eth_xy; ties to the lower row, then the lower column. NoC1 (the sender's NoC) routes -Y, then -X, wrapping around
    the grid, so the hop count is directional: a core one column left of a harvested column would wrap the whole row.
    Two links whose Ethernet cores sit over one live and one harvested column then get workers in two columns (as
    fabric_all_gather's C++ op places them), instead of both in the live column, where their writes to the routers
    would share the column's NoC1 segment into the Ethernet row."""
    t2p = _chip(mesh_device, coord)[1]
    gx, gy = _soc(mesh_device)[1]
    ex, ey = eth_xy
    best = None
    for x, y in sorted(allowed):
        if (x, y) in taken:
            continue
        lc = ttnn.CoreCoord(x, y)
        v = mesh_device.worker_core_from_logical_core(lc)
        px = t2p.get(v.x, v.x)
        score = ((v.y - ey) % gy + (px - ex) % gx, v.y, px)
        if best is None or score < best[0]:
            best = (score, lc)
    return best[1] if best is not None else None
