# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rms_norm against its torch semantics (reference.py), one random-input case per captured call
(cases.py). Math: PCC plus an elementwise atol/rtol bound, per device. The input differs per device (sharded over the
mesh columns) so each chip's output is checked on its own data; the weight is replicated."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_rms_norm_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    return p


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_rms_norm_ttnn(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(c["seed"])
    xs, ws = c["input"], c["weight"]

    x = torch.randn([n_dev, *xs["shape"][1:]], generator=g).to(torch.bfloat16)  # device d gets x[d:d+1]
    w = (1.0 + 0.5 * torch.randn(ws["shape"], generator=g)).to(torch.bfloat16)

    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        device=mesh_device,
        dtype=getattr(ttnn.DataType, xs["dtype"]),
        layout=getattr(ttnn.Layout, xs["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = ttnn.from_torch(
        w,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
        dtype=getattr(ttnn.DataType, ws["dtype"]),
        layout=getattr(ttnn.Layout, ws["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ck = c["compute_kernel_config"]
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, ck["math_fidelity"]),
        math_approx_mode=ck["math_approx_mode"],
        fp32_dest_acc_en=ck["fp32_dest_acc_en"],
        packer_l1_acc=ck["packer_l1_acc"],
        dst_full_sync_en=ck["dst_full_sync_en"],
    )

    out = ttnn.bringup.rms_norm(
        tt_x,
        epsilon=c["epsilon"],
        weight=tt_w,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=cfg,
    )
    outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == n_dev
    for d in range(n_dev):
        got = outs[d].reshape(xs["shape"]).double()
        want = ref.rms_norm(x[d : d + 1].reshape(xs["shape"]), epsilon=c["epsilon"], weight=w)
        pcc = _pcc(got, want)
        err = (got - want).abs()
        bound = c["atol"] + c["rtol"] * want.abs()
        n_bad = int((err > bound).sum())
        print(
            f"dev {d}: pcc {pcc:.7f} max abs err {float(err.max()):.4g} max rel {float((err / want.abs().clamp_min(1e-3)).max()):.4g}"
        )
        assert pcc >= c["pcc"], f"dev {d}: pcc {pcc} < {c['pcc']}"
        assert n_bad == 0, f"dev {d}: {n_bad} elements outside atol {c['atol']} + rtol {c['rtol']}"
