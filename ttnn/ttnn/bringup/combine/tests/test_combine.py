# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.combine against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Pure data movement with init_zeros: the whole output [1, 1, S, K, H] is compared exactly, the routed
(token, slot) pairs against their buffer rows and every other pair against 0. The input buffer is random everywhere,
padding rows included, so a read from a wrong row shows. A BFLOAT8_B buffer is compared against its own rounded
values (read back from the device). A dispatch group of several devices (the mesh rows, fabric on) sends every
routed row back to its source device of the group; _combine_groups."""

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
    # A case runs on a box of its own mesh size only (conftest skips it elsewhere): a smaller mesh opened on a bigger
    # box fails the FABRIC_2D router handshake (e.g. a 2x2 case on a 4x2 box), and the case's math depends on its mesh.
    p["require_exact_physical_num_devices"] = True
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


def _to_devices(mesh, t, spec):
    """t [n_dev, ...]: device d (row-major, d = r * cols + c) gets t[d:d+1]."""
    return ttnn.from_torch(
        t,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        device=mesh,
        dtype=getattr(ttnn.DataType, spec["dtype"]),
        layout=getattr(ttnn.Layout, spec["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _combine_groups(mesh_device, c):
    """Dispatch groups of dispatch_group_size = mesh rows devices (cluster_axis 0, fabric on): the mesh columns are
    the groups. Chip (j, col) holds the expert outputs of its 64 experts for tokens of both source devices of its
    column; combine sends each routed row back to its source device (r, col) at (t, k). With init_zeros the (t, k)
    routed to the other column's experts read 0. The whole output of every device is compared exactly."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    assert c["cluster_axis"] == 0 and c["dispatch_group_size"] == rows > 1 and c["init_zeros"]
    S, H, E, K, epc = (
        c["seq_len_per_chip"],
        c["emb_dim"],
        c["num_routed_experts"],
        c["num_experts_per_tok"],
        c["experts_per_chip"],
    )
    assert E == epc * n_dev
    N = c["max_dispatch_buffer_token_size"]
    g = torch.Generator().manual_seed(c["seed"])

    # Random expert outputs; metadata / counts / regions as dispatch + offset_cumsum leave them for random top-k ids.
    buf = torch.randn(n_dev, 1, N, H, generator=g).to(torch.bfloat16)
    table = ref.dispatch_table_groups(E, rows, cols)
    metas = torch.full((n_dev, 1, N, 3), -1, dtype=torch.int32)
    counts = torch.zeros(n_dev, E, dtype=torch.int64)
    regions = torch.zeros(n_dev, E, dtype=torch.int64)
    back = [[] for _ in range(n_dev)]  # per source device: (t, k, destination device, row)
    for col in range(cols):
        idx = torch.stack([ref.random_topk(S, E, K, g) for _ in range(rows)])  # the column's source devices
        offs, totals, reg = ref.group_routing(idx, table[col], epc)
        for j in range(rows):
            dst = j * cols + col
            counts[dst], regions[dst] = totals, reg
            for src, (t, k, e, row) in enumerate(ref.group_slots(idx, table[col], offs, j)):
                assert int(row.max()) < N
                metas[dst, 0, row, 0] = src * cols + col
                metas[dst, 0, row, 1] = t.to(torch.int32)
                metas[dst, 0, row, 2] = k.to(torch.int32)
                back[src * cols + col].append((t, k, dst, row))
    tt_buf = _to_devices(mesh_device, buf, c["buffer"])
    if c["buffer"]["dtype"] != "BFLOAT16":
        buf = torch.stack([ttnn.to_torch(t).reshape(1, N, H) for t in ttnn.get_device_tensors(tt_buf)]).to(
            torch.bfloat16
        )
    tt_meta = _to_devices(mesh_device, metas, c["metadata"])
    tt_cnt = _to_devices(mesh_device, counts.to(torch.int32), c["counts"])
    tt_reg = _to_devices(mesh_device, regions.to(torch.int32), c["regions"])
    # The device order the test assumes (row-major): device d holds counts[d].
    for d, t in enumerate(ttnn.get_device_tensors(tt_cnt)):
        assert torch.equal(ttnn.to_torch(t).reshape(-1).to(torch.int64), counts[d]), f"device order: dev {d}"

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
    outs = ttnn.get_device_tensors(out)
    assert len(outs) == n_dev
    for d, dt in enumerate(outs):
        got = ttnn.to_torch(dt)
        assert tuple(got.shape) == (1, 1, S, K, H), got.shape
        want = torch.zeros(S, K, H, dtype=torch.bfloat16)
        n = 0
        for t, k, dst, row in back[d]:
            want[t, k] = buf[dst, 0, row]
            n += t.numel()
        bad = (got.reshape(S, K, H) != want).any(dim=-1)
        print(f"dev {d}: {n} routed (token, slot) rows, {S * K - n} zero")
        assert not bad.any(), f"dev {d}: {int(bad.sum())}/{S * K} (token, slot) rows differ"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_combine(mesh_device, device_params, case):
    c = case
    if c["dispatch_group_size"] > 1:
        return _combine_groups(mesh_device, c)
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
