# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""[perf experiment fused_finish] determinism of the run-time fuse decision: on every perf / guard cell (1x4 ring
placement), the output with EVERY arrival entry forced late (FF_FORCE_LATE), forced fused (FF_FORCE_FUSED) and the
real dynamic decision (twice) must be bit-identical, and the base op's rel-RMS is printed alongside.
Run: scripts/run_safe_pytest.sh <this file>   (MRS_PERF_GUARD=1 adds the guard cells)"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
import ttnn

import test_ff_perf as H  # noqa: E402  (fixtures, cases, helpers)
from test_ff_perf import perf_mesh  # noqa: F401,E402


@pytest.mark.parametrize("case", H.CASES, ids=[H._id(c) for c in H.CASES])
def test_bitexact(perf_mesh, case):
    ex = case["extras"]
    a_shape, w_shape = case["inputs"]
    a = H.random_stacked(perf_mesh, a_shape, torch.bfloat16, seed=0)
    w = H.quantize_like_device(
        H.random_stacked(perf_mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5), case["weight_dtype"]
    )
    ta = H.create_ttnn_input_tensor(a, perf_mesh, dtype=case["dtype"], layout=ttnn.TILE_LAYOUT)
    tw = H.create_ttnn_input_tensor(w, perf_mesh, dtype=case["weight_dtype"], layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=ex["math_fidelity"], fp32_dest_acc_en=ex["fp32_dest_acc_en"])
    expected = H.pytorch_matmul_reduce_scatter(a, w, case["cluster_axis"], case["scatter_dim"])
    cols = tuple(perf_mesh.shape)[1]

    def run(ff, defines):
        os.environ["MRS_FF"] = ff
        os.environ["MRS_FF_DEFINES"] = defines
        out = H.matmul_reduce_scatter(
            ta,
            tw,
            cluster_axis=case["cluster_axis"],
            scatter_dim=case["scatter_dim"],
            topology=case["topology"],
            num_links=case["num_links"],
            compute_kernel_config=ckc,
        )
        ttnn.synchronize_device(perf_mesh)
        res = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        rms = max(
            (
                (g.double() - expected[i // cols, i % cols].double()).pow(2).mean().sqrt()
                / expected[i // cols, i % cols].double().pow(2).mean().sqrt()
            ).item()
            for i, g in enumerate(res)
        )
        return res, rms

    outs = {}
    opt = os.environ.get("FF_OPT", "")  # extra option defines under test (e.g. FF_HOLD_ACC)
    j = lambda *d: ",".join(x for x in (opt,) + d if x)
    for name, ff, d in [
        ("base", "base", ""),
        ("late", "5", j("FF_FORCE_LATE")),
        ("fused", "5", j("FF_FORCE_FUSED")),
        ("dyn1", "5", j()),
        ("dyn2", "5", j()),
    ]:
        outs[name] = run(ff, d)
    print(
        f"BITEXACT {H._id(case)} lag={H.mrs_ff._LAST_PLAN.get('lag_entries')} "
        + " ".join(f"{k}:rr={v[1]:.5f}" for k, v in outs.items())
    )
    for name in ("fused", "dyn1", "dyn2"):
        for i, (x, y) in enumerate(zip(outs["late"][0], outs[name][0])):
            nd = (x != y).sum().item()
            print(f"  late vs {name} dev {i}: {nd} differing elements")
            assert nd == 0, f"{name} differs from late on device {i}: {nd} elements"
    os.environ["MRS_FF_DEFINES"] = opt
    os.environ["MRS_FF"] = "5"
