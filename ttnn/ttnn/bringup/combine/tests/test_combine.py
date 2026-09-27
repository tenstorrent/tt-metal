# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.combine against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Pure data movement with init_zeros: the whole output [1, 1, S, K, H] is compared exactly, the routed
(token, slot) pairs against their buffer rows and every other pair against 0. The input buffer is random everywhere,
padding rows included, so a read from a wrong row shows. A BFLOAT8_B buffer is compared against its own rounded
values (read back from the device)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_combine_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    return p


def _mem(name):
    return {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[name]


def _to_mesh(mesh, t, spec):
    """t [cols * n0, ...]: device (0, col) gets rows col*n0..(col+1)*n0 (a 1-row mesh; per-device data differs)."""
    return ttnn.from_torch(
        t,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(None, 0)),
        device=mesh,
        dtype=getattr(ttnn.DataType, spec["dtype"]),
        layout=getattr(ttnn.Layout, spec["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_combine(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    assert rows == 1 and c["dispatch_group_size"] == 1, "reference covers a 1-device dispatch axis (mesh rows)"
    S, H, E, K, epc = (
        c["seq_len_per_chip"],
        c["emb_dim"],
        c["num_routed_experts"],
        c["num_experts_per_tok"],
        c["experts_per_chip"],
    )
    N = c["max_dispatch_buffer_token_size"]
    g = torch.Generator().manual_seed(c["seed"])

    # Random expert outputs, and the metadata / counts / regions dispatch + offset_cumsum give for random top-k ids.
    buf = torch.randn(cols, 1, N, H, generator=g).to(torch.bfloat16)
    table = ref.dispatch_table(E, cols)
    metas, counts, regions, routed = [], [], [], []
    for d in range(cols):
        idx = ref.random_topk(S, E, K, g)
        off, cnt, reg = ref.offsets_counts_regions(idx, table[d], epc)
        m, t, k, row = ref.metadata(idx, table[d], off, N, group=d, num_groups=cols)
        assert int(row.max()) < N
        metas.append(m.reshape(1, N, 3))
        counts.append(cnt.reshape(1, E))
        regions.append(reg.reshape(1, E))
        routed.append((t, k, row))
    tt_buf = _to_mesh(mesh_device, buf, c["buffer"])
    if c["buffer"]["dtype"] != "BFLOAT16":
        # A block-float buffer (BFLOAT8_B): the device holds the rounded values, and combine unpacks them to bf16
        # losslessly (a bfp8 mantissa fits bf16), so the exact reference is the buffer as the device stores it.
        buf = torch.stack([ttnn.to_torch(t).reshape(1, N, H) for t in ttnn.get_device_tensors(tt_buf)]).to(
            torch.bfloat16
        )
    tt_meta = _to_mesh(mesh_device, torch.stack(metas), c["metadata"])
    tt_cnt = _to_mesh(mesh_device, torch.cat(counts).to(torch.int32), c["counts"])
    tt_reg = _to_mesh(mesh_device, torch.cat(regions).to(torch.int32), c["regions"])

    out = ttnn.bringup.combine(
        tt_buf,
        tt_meta,
        tt_cnt,
        tt_reg,
        dispatch_group_size=c["dispatch_group_size"],
        experts_per_chip=epc,
        num_experts_per_tok=K,
        seq_len_per_chip=S,
        cluster_axis=c["cluster_axis"],
        num_links=c["num_links"],
        topology=getattr(ttnn.Topology, c["topology"]),
        memory_config=_mem(c["memory_config"]),
        init_zeros=c["init_zeros"],
        use_fp8_combine=c["use_fp8_combine"],
    )
    for d, dt in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(dt)
        assert tuple(got.shape) == (1, 1, S, K, H), got.shape
        want = ref.combine(buf[d, 0], *routed[d], seq=S, top_k=K, init_zeros=c["init_zeros"])
        bad = (got.reshape(S, K, H) != want).any(dim=-1)
        assert not bad.any(), f"dev {d}: {int(bad.sum())}/{S * K} (token, slot) rows differ"
