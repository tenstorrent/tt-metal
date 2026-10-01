# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.dispatch with allow_cluster_axis_1=True: one dispatch group of 4 chips along mesh axis 1 (1x4 mesh,
FABRIC_2D). Chip j holds experts j*epc .. (j+1)*epc - 1; every source chip r sends its (token, slot) pairs to the chip
of their expert. Every row a chip receives is compared exactly (buffer and metadata [r, t, k]; on a 1x4 mesh the
source's linearized coordinate is its column r). Also: the default (option off) still refuses cluster_axis=1."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_spec = importlib.util.spec_from_file_location("bringup_dispatch_unit_ref", Path(__file__).parents[1] / "reference.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 24576}


def _to_devices(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        device=mesh,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _inputs(mesh, S, H, E, K, seed):
    n = mesh.get_num_devices()
    epc = E // n
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, S, H, generator=g).to(torch.bfloat16)
    idx = torch.stack([ref.random_topk(S, E, K, g) for _ in range(n)])  # [n, S, K]
    table = ref.dispatch_table_groups(E, n, 1)[0]  # [E + 1]: expert e -> chip e // epc
    offs, totals, regions = ref.group_routing(idx, table, epc)
    N = n * S * 8 + ref.TILE * (epc - 1)  # compute_constants, capacity factor 8
    tt = dict(
        input_tensor=_to_devices(mesh, x, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        indices_tensor=_to_devices(mesh, idx.to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
        expert_offsets_tensor=_to_devices(mesh, offs.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        expert_dispatch_table_tensor=_to_devices(
            mesh, table.unsqueeze(0).expand(n, -1).contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        ),
    )
    kw = dict(
        dispatch_group_size=n,
        experts_per_chip=epc,
        num_routed_experts=E,
        num_experts_per_tok=K,
        metadata_len=3,
        max_dispatch_buffer_token_size=N,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )
    return x, idx, table, offs, N, tt, kw


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("S, H, E, K", [(256, 1024, 256, 8), (1280, 4096, 256, 8)], ids=["s256-h1024", "s1280-h4096"])
def test_dispatch_cluster_axis_1(mesh_device, device_params, S, H, E, K):
    n = mesh_device.get_num_devices()
    x, idx, table, offs, N, tt, kw = _inputs(mesh_device, S, H, E, K, seed=0)
    buf, meta = ttnn.bringup.dispatch(**tt, **kw, allow_cluster_axis_1=True)
    bufs = [ttnn.to_torch(d).reshape(N, H) for d in ttnn.get_device_tensors(buf)]
    metas = [ttnn.to_torch(d).reshape(N, 3).to(torch.int64) for d in ttnn.get_device_tensors(meta)]
    total = 0
    for j in range(n):
        for src, (t, k, e, row) in enumerate(ref.group_slots(idx, table, offs, j)):
            assert row.numel() > 0 and int(row.max()) < N
            total += row.numel()
            want_m = torch.stack([torch.full_like(t, src), t, k], dim=1)
            bad_m = (metas[j][row] != want_m).any(dim=1)
            assert not bad_m.any(), f"chip {j} from {src}: {int(bad_m.sum())}/{row.numel()} metadata rows differ"
            bad_b = (bufs[j][row] != x[src][t]).any(dim=1)
            assert not bad_b.any(), f"chip {j} from {src}: {int(bad_b.sum())}/{row.numel()} buffer rows differ"
    assert total == n * S * K, f"{total} dispatched rows, want {n * S * K} (every expert lives in the one group)"


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_dispatch_cluster_axis_1_refused_by_default(mesh_device, device_params, expect_error):
    _, _, _, _, _, tt, kw = _inputs(mesh_device, 32, 256, 256, 8, seed=1)
    with expect_error(RuntimeError, "cluster_axis must be 0"):
        ttnn.bringup.dispatch(**tt, **kw)
