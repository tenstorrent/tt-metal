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
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh.shape, dims=(None, 0))


def _to_mesh(mesh, t, spec):
    """t [cols * n0, ...]: device (0, col) gets rows col*n0..(col+1)*n0 (a 1-row mesh; per-device data differs)."""
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
    assert rows == 1, "inputs are laid out per mesh column (1-row mesh)"
    N, H, I = c["buffer_rows"], c["emb_dim"], c["hidden_dim"]
    E, epc, S, K = c["num_routed_experts"], c["experts_per_chip"], c["seq_len_per_chip"], c["num_experts_per_tok"]
    assert E == epc * cols
    g = torch.Generator().manual_seed(c["seed"])

    # Counts / regions from random top-k ids by the dispatch rules (chip d holds experts d*epc .. d*epc + epc - 1).
    table = ref.dispatch_table(E, cols)
    counts, regions = [], []
    for d in range(cols):
        _, cnt, reg = ref.offsets_counts_regions(ref.random_topk(S, E, K, g), table[d], epc)
        assert int(cnt.max()) <= c["max_dispatched_tokens_per_expert"] and int((reg + cnt).max()) <= N
        counts.append(cnt)
        regions.append(reg)
    gidx = torch.arange(E, dtype=torch.int32)  # chip d, local slot le -> global expert d*epc + le

    # Random dispatched buffer, random everywhere (padding rows too).
    x = torch.randn(cols * N, H, generator=g).to(torch.bfloat16)

    # Weights ~ N(0, 1/fan_in), rounded to bfp8 on the host; the reference uses the rounded values.
    wspec = c["weights"]
    wdt, wlay = getattr(ttnn.DataType, wspec["dtype"]), getattr(ttnn.Layout, wspec["layout"])
    compose = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    want = [torch.zeros(N, H) for _ in range(cols)]
    tt_w = {"gate": [], "up": [], "down": []}
    for le in range(epc):
        host = {}
        for name, shape, fan_in in (("gate", (H, I), H), ("up", (H, I), H), ("down", (I, H), I)):
            w = torch.randn(cols * shape[0], shape[1], generator=g) * fan_in**-0.5
            t = ttnn.from_torch(w, dtype=wdt, layout=wlay, mesh_mapper=_shard(mesh_device))
            host[name] = ttnn.to_torch(t, mesh_composer=compose).reshape(cols, *shape)
            tt_w[name].append(ttnn.to_device(t, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        for d in range(cols):
            e = d * epc + le
            r, n = int(regions[d][e]), int(counts[d][e])
            if n:
                xe = x[d * N + r : d * N + r + n]
                want[d][r : r + n] = ref.expert_ffn(
                    xe, host["gate"][d], host["up"][d], host["down"][d], c["activation"]
                )

    tt_x = _to_mesh(mesh_device, x, c["buffer"])
    tt_reg = _to_mesh(mesh_device, torch.stack(regions).to(torch.int32), {"dtype": "UINT32", "layout": "ROW_MAJOR"})
    tt_cnt = _to_mesh(mesh_device, torch.stack(counts).to(torch.int32), {"dtype": "UINT32", "layout": "ROW_MAJOR"})
    tt_gidx = _to_mesh(mesh_device, gidx, {"dtype": "UINT32", "layout": "ROW_MAJOR"})
    assert list(tt_reg.shape) == [1, E] and list(tt_gidx.shape) == [epc] and list(tt_x.shape) == [N, H]

    out = ttnn.bringup.unified_routed_expert_moe(
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
    for d, dt in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(dt).float().reshape(N, H)
        routed = torch.zeros(N, dtype=torch.bool)
        for le in range(epc):
            e = d * epc + le
            r, n = int(regions[d][e]), int(counts[d][e])
            routed[r : r + n] = True
        a, b = got[routed], want[d][routed]
        pcc = _pcc(a, b)
        rel = float((a - b).norm() / b.norm())
        logger.info(f"dev {d}: {int(routed.sum())} routed rows, pcc {pcc:.6f}, rel {rel:.5f}")
        assert (
            pcc >= c["pcc"] and rel <= c["rel"]
        ), f"dev {d}: pcc {pcc:.6f} (min {c['pcc']}), rel {rel:.5f} (max {c['rel']})"
