# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.combine with allow_cluster_axis_1=True: one combine group of 4 chips along mesh axis 1 (1x4 mesh,
FABRIC_2D). Chip j holds the expert outputs of experts j*epc .. (j+1)*epc - 1 for the tokens of every source chip r
(metadata [r, t, k] as dispatch leaves it); combine sends each row back to chip r at (t, k). Every (token, slot) of
every chip is compared exactly. Also: the default (option off) still refuses cluster_axis=1."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_spec = importlib.util.spec_from_file_location("bringup_combine_unit_ref", Path(__file__).parents[1] / "reference.py")
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
    N = n * S * 8 + ref.TILE * (epc - 1)  # compute_constants, capacity factor 8
    g = torch.Generator().manual_seed(seed)
    buf = torch.randn(n, 1, N, H, generator=g).to(torch.bfloat16)
    idx = torch.stack([ref.random_topk(S, E, K, g) for _ in range(n)])  # [n, S, K]
    table = ref.dispatch_table_groups(E, n, 1)[0]
    offs, totals, regions = ref.group_routing(idx, table, epc)
    metas = torch.full((n, 1, N, 3), -1, dtype=torch.int32)
    want = torch.zeros(n, S, K, H, dtype=torch.bfloat16)
    for j in range(n):
        for src, (t, k, e, row) in enumerate(ref.group_slots(idx, table, offs, j)):
            assert int(row.max()) < N
            metas[j, 0, row, 0] = src
            metas[j, 0, row, 1] = t.to(torch.int32)
            metas[j, 0, row, 2] = k.to(torch.int32)
            want[src, t, k] = buf[j, 0, row]
    rep = lambda v: v.unsqueeze(0).expand(n, -1).contiguous().to(torch.int32)
    tt = (
        _to_devices(mesh, buf, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        _to_devices(mesh, metas, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        _to_devices(mesh, rep(totals), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        _to_devices(mesh, rep(regions), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )
    kw = dict(
        dispatch_group_size=n,
        experts_per_chip=epc,
        num_experts_per_tok=K,
        seq_len_per_chip=S,
        cluster_axis=1,
        num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        init_zeros=True,
    )
    return want, tt, kw


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("S, H, E, K", [(256, 1024, 256, 8), (1280, 4096, 256, 8)], ids=["s256-h1024", "s1280-h4096"])
def test_combine_cluster_axis_1(mesh_device, device_params, S, H, E, K):
    want, tt, kw = _inputs(mesh_device, S, H, E, K, seed=0)
    out = ttnn.bringup.combine(*tt, **kw, allow_cluster_axis_1=True)
    outs = ttnn.get_device_tensors(out)
    assert len(outs) == mesh_device.get_num_devices()
    for r, dt in enumerate(outs):
        got = ttnn.to_torch(dt)
        assert tuple(got.shape) == (1, 1, S, kw["num_experts_per_tok"], want.shape[-1]), got.shape
        bad = (got.reshape(want.shape[1:]) != want[r]).any(dim=-1)
        assert not bad.any(), f"chip {r}: {int(bad.sum())}/{bad.numel()} (token, slot) rows differ"


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_combine_cluster_axis_1_refused_by_default(mesh_device, device_params, expect_error):
    _, tt, kw = _inputs(mesh_device, 32, 256, 256, 8, seed=1)
    with expect_error(RuntimeError, "cluster_axis must be 0"):
        ttnn.bringup.combine(*tt, **kw)
