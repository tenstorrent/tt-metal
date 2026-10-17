# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the l1-mode mesh-ring hang. HR_CALLS calls, each synchronized and logged (which call hangs), HR_ROWS rows
per chip (default 4096), HR_TOPO line|ring, HR_AXIS (none, 0, 1). RS_PROBE=<path> for the triage dump."""

import os
import time

import pytest
import torch

import ttnn

ROWS = int(os.environ.get("HR_ROWS", "4096"))
CALLS = int(os.environ.get("HR_CALLS", "5"))
TOPO = ttnn.Topology.Ring if os.environ.get("HR_TOPO", "ring") == "ring" else ttnn.Topology.Linear
AXIS = None if os.environ.get("HR_AXIS", "none") == "none" else int(os.environ["HR_AXIS"])
MODE = os.environ.get("RS_PORT_MODE", "l1")
H = 4096


def _device_params():
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = 8192
    return {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "fabric_router_config": rc}


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True, ids=["2x4"])
def test_rs_hang_repro(mesh_device):
    from ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter import fabric_reduce_scatter

    rows, cols = tuple(mesh_device.shape)
    G = rows * cols if AXIS is None else (rows, cols)[AXIS]
    T = ROWS - ROWS % (32 * G)
    torch.manual_seed(0)
    parts = torch.randn(rows, cols, T, H).bfloat16()
    x = ttnn.from_torch(
        parts,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    x = ttnn.reshape(x, (1, 1, T, H))
    y = None
    for i in range(CALLS):
        t0 = time.time()
        print(f"HANG_REPRO call {i} start", flush=True)
        y = fabric_reduce_scatter(x, cluster_axis=AXIS, topology=TOPO, num_links=2, output=y, port_mode=MODE)
        ttnn.synchronize_device(mesh_device)
        print(f"HANG_REPRO call {i} done {time.time() - t0:.3f} s", flush=True)
