# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.dispatch against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Pure data movement: every row the op writes (one per (token, top-k slot) routed to a present expert) is
compared exactly, buffer and metadata; with a dispatch group of several devices (the mesh rows) every row a chip
receives from any source device of its group. Don't-care: buffer / metadata rows no (token, slot) lands on (tile padding
between expert regions and the unused tail)."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_dispatch_tests_{name}", _HERE / f"{name}.py")
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


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


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


def _dispatch_groups(mesh_device, c):
    """Dispatch groups of dispatch_group_size = mesh rows devices (cluster_axis 0, fabric on): the mesh columns are
    the groups. Every row a destination chip receives from any source device of its column is compared exactly."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    assert c["cluster_axis"] == 0 and c["dispatch_group_size"] == rows > 1
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

    # Random inputs per device d = r * cols + col; offsets from offset_cumsum's rules over each column's group.
    x = torch.randn(rows, cols, S, H, generator=g).to(torch.bfloat16)
    idx = torch.stack([torch.stack([ref.random_topk(S, E, K, g) for _ in range(cols)]) for _ in range(rows)])
    table = ref.dispatch_table_groups(E, rows, cols)  # [cols, E + 1]
    offs = torch.zeros(rows, cols, E, dtype=torch.int64)
    for col in range(cols):
        offs[:, col] = ref.group_routing(idx[:, col], table[col], epc)[0]

    tt_x = _to_devices(mesh_device, x.reshape(n_dev, S, H), c["input"])
    tt_idx = _to_devices(mesh_device, idx.reshape(n_dev, S, K).to(torch.int32), c["indices"])
    tt_off = _to_devices(mesh_device, offs.reshape(n_dev, E).to(torch.int32), c["offsets"])
    tab_dev = table.unsqueeze(0).expand(rows, cols, E + 1).reshape(n_dev, E + 1)
    tt_tab = _to_devices(mesh_device, tab_dev.contiguous(), c["table"])
    # The device order the test assumes (row-major): device d holds the table row of its column d % cols.
    for d, t in enumerate(_per_device(tt_tab)):
        assert torch.equal(t.reshape(-1).to(torch.int32), table[d % cols]), f"device order: dev {d}"

    buf, meta = ttnn.bringup.dispatch(
        input_tensor=tt_x,
        indices_tensor=tt_idx,
        expert_offsets_tensor=tt_off,
        expert_dispatch_table_tensor=tt_tab,
        dispatch_group_size=c["dispatch_group_size"],
        experts_per_chip=epc,
        num_routed_experts=E,
        num_experts_per_tok=K,
        metadata_len=c["metadata_len"],
        max_dispatch_buffer_token_size=N,
        cluster_axis=c["cluster_axis"],
        num_links=c["num_links"],
        topology=getattr(ttnn.Topology, c["topology"]),
        fp8_output=c["fp8_output"],
        subdevice_id=c["subdevice_id"],
        num_workers_per_sender=c["num_workers_per_sender"],
    )
    bufs, metas = _per_device(buf), _per_device(meta)
    assert len(bufs) == n_dev
    for d in range(n_dev):
        j, col = divmod(d, cols)  # destination chip j of group col
        b = bufs[d].reshape(N, H)
        m = metas[d].reshape(N, c["metadata_len"]).to(torch.int64)
        n_rows = 0
        for src, (t, k, e, row) in enumerate(ref.group_slots(idx[:, col], table[col], offs[:, col], j)):
            assert row.numel() > 0 and int(row.max()) < N
            n_rows += row.numel()
            want_m = torch.stack([torch.full_like(t, src * cols + col), t, k], dim=1)
            bad_m = (m[row, :3] != want_m).any(dim=1)
            assert not bad_m.any(), f"dev {d} from row {src}: {int(bad_m.sum())}/{row.numel()} metadata rows differ"
            bad_b = (b[row] != x[src, col][t]).any(dim=1)
            assert not bad_b.any(), f"dev {d} from row {src}: {int(bad_b.sum())}/{row.numel()} buffer rows differ"
        print(f"dev {d}: {n_rows} dispatched rows exact")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_dispatch(mesh_device, device_params, case):
    c = case
    if c["dispatch_group_size"] > 1:
        return _dispatch_groups(mesh_device, c)
    rows, cols = c["mesh"]
    assert rows == 1 and c["dispatch_group_size"] == 1, "reference covers a 1-device dispatch axis (mesh rows)"
    S, H, E, K, epc = (
        c["seq_len_per_chip"],
        c["emb_dim"],
        c["num_routed_experts"],
        c["num_experts_per_tok"],
        c["experts_per_chip"],
    )
    assert E == epc * cols
    g = torch.Generator().manual_seed(c["seed"])

    # Random inputs per device: x, top-k ids (distinct per token); offsets / table built from them by the op's rules.
    x = torch.randn(cols, S, H, generator=g).to(torch.bfloat16)
    idx = torch.stack([ref.random_topk(S, E, K, g) for _ in range(cols)])  # [cols, S, K]
    table = ref.dispatch_table(E, cols)  # [cols, E + 1]
    offs = torch.stack([ref.offsets_counts_regions(idx[d], table[d], epc)[0] for d in range(cols)])  # [cols, E]

    tt_x = _to_mesh(mesh_device, x, c["input"])
    tt_idx = _to_mesh(mesh_device, idx.to(torch.int32), c["indices"])
    tt_off = _to_mesh(mesh_device, offs.to(torch.int32), c["offsets"])
    tt_tab = _to_mesh(mesh_device, table, c["table"])

    buf, meta = ttnn.bringup.dispatch(
        input_tensor=tt_x,
        indices_tensor=tt_idx,
        expert_offsets_tensor=tt_off,
        expert_dispatch_table_tensor=tt_tab,
        dispatch_group_size=c["dispatch_group_size"],
        experts_per_chip=epc,
        num_routed_experts=E,
        num_experts_per_tok=K,
        metadata_len=c["metadata_len"],
        max_dispatch_buffer_token_size=c["max_dispatch_buffer_token_size"],
        cluster_axis=c["cluster_axis"],
        num_links=c["num_links"],
        topology=getattr(ttnn.Topology, c["topology"]),
        fp8_output=c["fp8_output"],
        subdevice_id=c["subdevice_id"],
        num_workers_per_sender=c["num_workers_per_sender"],
    )
    bufs, metas = _per_device(buf), _per_device(meta)
    N = c["max_dispatch_buffer_token_size"]
    for d in range(cols):
        b = bufs[d].reshape(N, H)
        m = metas[d].reshape(N, c["metadata_len"]).to(torch.int64)
        rows_d, want_b, want_m = ref.dispatch(x[d], idx[d], offs[d].long(), table[d], group=d, num_groups=cols)
        assert rows_d.numel() > 0 and int(rows_d.max()) < N
        got_m = m[rows_d, :3]
        bad_m = (got_m != want_m).any(dim=1)
        assert not bad_m.any(), f"dev {d}: {int(bad_m.sum())}/{rows_d.numel()} metadata rows differ"
        bad_b = (b[rows_d] != want_b).any(dim=1)
        assert not bad_b.any(), f"dev {d}: {int(bad_b.sum())}/{rows_d.numel()} buffer rows differ"
