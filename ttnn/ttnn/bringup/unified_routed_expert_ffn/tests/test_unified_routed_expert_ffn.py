# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.unified_routed_expert_moe against its torch semantics (reference.py), one random-input case per
captured call (cases.py). Math: over every routed row of every chip (the rows [region, region + count) of each local
expert), PCC and the relative Frobenius error against a float32 reference on the same bf16 x and bfp8-rounded weights,
per chip. Don't-care: the output rows outside the experts' routed rows (tile padding, the unused tail)."""

import importlib.util
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_urffn_tests_{name}", _HERE / f"{name}.py")
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


def _kernel_config(k):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, k["math_fidelity"]),
        math_approx_mode=k["math_approx_mode"],
        fp32_dest_acc_en=k["fp32_dest_acc_en"],
        packer_l1_acc=k["packer_l1_acc"],
        dst_full_sync_en=k["dst_full_sync_en"],
    )


def _shard(mesh):
    """Shard dim 0 over every device, row-major: device (r, c) = index d = r * cols + c gets block d."""
    if tuple(mesh.shape)[0] == 1:
        return ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(None, 0))
    return ttnn.ShardTensorToMesh(mesh, dim=0)


def _experts_of(d, rows, cols, epc):
    """(chip experts, dispatch-group experts) of device d = r * cols + c, ExpertMapping col-major: the group is mesh
    column c (its rows chips), chip (r, c) holds experts (c * rows + r) * epc .. + epc - 1. A 1-row mesh: chip c
    holds c * epc .. + epc - 1 and is its own group."""
    r, c = divmod(d, cols)
    chip = (c * rows + r) * epc + torch.arange(epc)
    group = c * rows * epc + torch.arange(rows * epc)
    return chip, group


def _to_mesh(mesh, t, spec):
    """t [n_dev * n0, ...]: device d (row-major) gets rows d*n0..(d+1)*n0 (per-device data differs)."""
    return ttnn.from_torch(
        t,
        mesh_mapper=_shard(mesh),
        device=mesh,
        dtype=getattr(ttnn.DataType, spec["dtype"]),
        layout=getattr(ttnn.Layout, spec["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert_moe(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    N, H, I = c["buffer_rows"], c["emb_dim"], c["hidden_dim"]
    E, epc, S, K = c["num_routed_experts"], c["experts_per_chip"], c["seq_len_per_chip"], c["num_experts_per_tok"]
    assert E == epc * n_dev
    g = torch.Generator().manual_seed(c["seed"])

    chip_experts = [_experts_of(d, rows, cols, epc)[0] for d in range(n_dev)]
    counts, regions, x = _routed_inputs(c, rows, cols, g)
    gidx = torch.cat(chip_experts).to(torch.int32)  # device d, local slot le -> global expert chip_experts[d][le]

    # Weights ~ N(0, 1/fan_in) (gate / up times gate_up_scale when set), rounded to bfp8 on the host; the reference
    # uses the rounded values.
    wspec = c["weights"]
    wdt, wlay = getattr(ttnn.DataType, wspec["dtype"]), getattr(ttnn.Layout, wspec["layout"])
    compose = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    want = [torch.zeros(N, H) for _ in range(n_dev)]
    tt_w = {"gate": [], "up": [], "down": []}
    for le in range(epc):
        host = {}
        for name, shape, fan_in in (("gate", (H, I), H), ("up", (H, I), H), ("down", (I, H), I)):
            w = torch.randn(n_dev * shape[0], shape[1], generator=g) * fan_in**-0.5
            if name != "down" and c.get("gate_up_scale"):  # push gate / up into an activation's clamp range
                w = w * c["gate_up_scale"]
            t = ttnn.from_torch(w, dtype=wdt, layout=wlay, mesh_mapper=_shard(mesh_device))
            host[name] = ttnn.to_torch(t, mesh_composer=compose).reshape(n_dev, *shape)
            tt_w[name].append(ttnn.to_device(t, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        for d in range(n_dev):
            e = int(chip_experts[d][le])
            r, n = int(regions[d][e]), int(counts[d][e])
            if n:
                xe = x[d * N + r : d * N + r + n]
                want[d][r : r + n] = ref.expert_ffn(
                    xe, host["gate"][d], host["up"][d], host["down"][d], c["activation"]
                )

    dev = _device_inputs(mesh_device, c, counts, regions, x, gidx)
    out = _call(c, dev, tt_w)
    masks = []
    for d, dt in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(dt).float().reshape(N, H)
        routed = torch.zeros(N, dtype=torch.bool)
        for le in range(epc):
            e = int(chip_experts[d][le])
            r, n = int(regions[d][e]), int(counts[d][e])
            routed[r : r + n] = True
        masks.append(routed)
        a, b = got[routed], want[d][routed]
        pcc = _pcc(a, b)
        rel = float((a - b).norm() / b.norm())
        logger.info(f"dev {d}: {int(routed.sum())} routed rows, pcc {pcc:.6f}, rel {rel:.5f}")
        assert (
            pcc >= c["pcc"] and rel <= c["rel"]
        ), f"dev {d}: pcc {pcc:.6f} (min {c['pcc']}), rel {rel:.5f} (max {c['rel']})"

    # B: other routing and buffer (seed + 1), the same weights (they are the op's parameters, and the costly part).
    counts_b, regions_b, x_b = _routed_inputs(c, rows, cols, torch.Generator().manual_seed(c["seed"] + 1))
    dev_b = _device_inputs(mesh_device, c, counts_b, regions_b, x_b, gidx)
    # Compared: A's routed rows (the rows outside them are don't-care, left as the allocator hands them over).
    determinism.assert_deterministic(
        lambda: _routed_rows(c, _call(c, dev, tt_w), masks),
        lambda: _call(c, dev_b, tt_w),
        first=_routed_rows(c, out, masks),
        label=c["id"],
    )


def _routed_rows(c, out, masks):
    N, H = c["buffer_rows"], c["emb_dim"]
    return [ttnn.to_torch(t).reshape(N, H)[m] for t, m in zip(ttnn.get_device_tensors(out), masks)]


def _routed_inputs(c, rows, cols, g):
    """Per-device counts and regions from random top-k ids by the dispatch rules, and the random dispatched buffer."""
    n_dev = rows * cols
    N, H = c["buffer_rows"], c["emb_dim"]
    E, epc, S, K = c["num_routed_experts"], c["experts_per_chip"], c["seq_len_per_chip"], c["num_experts_per_tok"]
    # Counts / regions from random top-k ids by the dispatch rules: counts over the device's dispatch group (the
    # experts of its mesh column; the chip itself on a 1-row mesh), regions per chip (_experts_of).
    counts, regions = [], []
    for d in range(n_dev):
        table_row = torch.full((E + 1,), -1, dtype=torch.int32)
        table_row[_experts_of(d, rows, cols, epc)[1]] = 0
        _, cnt, reg = ref.offsets_counts_regions(ref.random_topk(S, E, K, g), table_row, epc)
        assert int(cnt.max()) <= c["max_dispatched_tokens_per_expert"] and int((reg + cnt).max()) <= N
        counts.append(cnt)
        regions.append(reg)
    # Random dispatched buffer, random everywhere (padding rows too). x_scale (default 1) widens it so a clamped
    # activation's limit is reached (the projections are ~N(0, x_scale^2)).
    x = (torch.randn(n_dev * N, H, generator=g) * c.get("x_scale", 1.0)).to(torch.bfloat16)
    return counts, regions, x


def _device_inputs(mesh_device, c, counts, regions, x, gidx):
    E, epc = c["num_routed_experts"], c["experts_per_chip"]
    N, H = c["buffer_rows"], c["emb_dim"]
    tt_x = _to_mesh(mesh_device, x, c["buffer"])
    tt_reg = _to_mesh(mesh_device, torch.stack(regions).to(torch.int32), {"dtype": "UINT32", "layout": "ROW_MAJOR"})
    tt_cnt = _to_mesh(mesh_device, torch.stack(counts).to(torch.int32), {"dtype": "UINT32", "layout": "ROW_MAJOR"})
    tt_gidx = _to_mesh(mesh_device, gidx, {"dtype": "UINT32", "layout": "ROW_MAJOR"})
    assert list(tt_reg.shape) == [1, E] and list(tt_gidx.shape) == [epc] and list(tt_x.shape) == [N, H]
    return tt_x, tt_reg, tt_cnt, tt_gidx


def _call(c, dev, tt_w):
    tt_x, tt_reg, tt_cnt, tt_gidx = dev
    return ttnn.bringup.unified_routed_expert_moe(
        tt_x,
        tt_reg,
        tt_cnt,
        tt_gidx,
        tt_w["gate"],
        tt_w["up"],
        tt_w["down"],
        max_dispatched_tokens_per_expert=c["max_dispatched_tokens_per_expert"],
        compute_kernel_config=_kernel_config(c["compute_kernel_config"]),
        activation=getattr(ttnn.bringup.RoutedExpertActivation, c["activation"]),
        high_precision=c["high_precision"],
    )
