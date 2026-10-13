# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.fabric_reduce_scatter against its torch semantics (reference.py), one random-input case per captured
call (cases.py). Math (a sum of bf16 partials, added in bf16 hop by hop): per chip, PCC, the relative L2 error and
an elementwise atol + rtol bound against a float64 sum of the same bf16 inputs. Every chip's input differs, so a chip given the
wrong block or the wrong group's sum fails. Don't-care: nothing.

The op's own unit suite is tests/unit/."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"bringup_fabric_reduce_scatter_ttnn_tests_{name}", _HERE / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    # A case runs on a box of its own mesh size only: a smaller mesh opened on a bigger box fails the FABRIC_2D router
    # handshake, and the groups depend on the mesh.
    p["require_exact_physical_num_devices"] = True
    return p


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _inputs(mesh_device, c, seed):
    """Host per-device inputs (row-major list) and the device input (chip (r, c) holds xs[r * cols + c])."""
    rows, cols = c["mesh"]
    g = torch.Generator().manual_seed(seed)
    _, _, n, w = c["input"]["shape"]
    host = torch.randn(rows, cols, n, w, generator=g).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=getattr(ttnn.DataType, c["input"]["dtype"]),
        layout=getattr(ttnn.Layout, c["input"]["layout"]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    assert list(tt_in.shape) == c["input"]["shape"]
    return [host[r, cc].reshape(c["input"]["shape"]) for r in range(rows) for cc in range(cols)], tt_in


def _call(c, tt_in):
    kw = {"topology": getattr(ttnn.Topology, c["topology"])} if c["topology"] is not None else {}
    return ttnn.bringup.fabric_reduce_scatter(tt_in, cluster_axis=c["cluster_axis"], num_links=c["num_links"], **kw)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_fabric_reduce_scatter_ttnn(mesh_device, device_params, case):
    c = case
    rows, cols = c["mesh"]
    xs, tt_in = _inputs(mesh_device, c, c["seed"])
    out = _call(c, tt_in)
    outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    assert len(outs) == rows * cols
    for d in range(rows * cols):
        want = ref.reduce_scatter(xs, (rows, cols), d, cluster_axis=c["cluster_axis"])
        got = outs[d].reshape(want.shape).double()
        pcc = _pcc(got, want)
        err = (got - want).abs()
        rel = float((got - want).norm() / want.norm())
        n_bad = int((err > c["atol"] + c["rtol"] * want.abs()).sum())
        print(
            f"dev {d}: pcc {pcc:.7f} rel {rel:.3e} max abs err {float(err.max()):.4g} "
            f"max err - rtol |want| {float((err - c['rtol'] * want.abs()).max()):.4g}"
        )
        assert pcc >= c["pcc"] and rel <= c["rel"], f"dev {d}: pcc {pcc} (min {c['pcc']}), rel {rel} (max {c['rel']})"
        assert n_bad == 0, f"dev {d}: {n_bad} elements outside atol {c['atol']} + rtol {c['rtol']}"
    _, tt_in_b = _inputs(mesh_device, c, c["seed"] + 1)
    determinism.assert_deterministic(lambda: _call(c, tt_in), lambda: _call(c, tt_in_b), first=out, label=c["id"])
