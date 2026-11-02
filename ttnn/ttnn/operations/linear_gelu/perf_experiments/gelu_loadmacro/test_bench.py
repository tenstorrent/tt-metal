# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""gelu_loadmacro isolated bench.

Precision: GV_VARIANTS=0,1 scripts/run_safe_pytest.sh <this>::test_precision
Perf:      GV_VARIANTS=0,1 scripts/run_safe_pytest.sh --profile <this>::test_perf
           (DEVICE KERNEL DURATION per launch from ops_perf_results, in the printed launch order)
"""
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

VARIANTS = [int(v) for v in os.environ.get("GV_VARIANTS", "0,1").split(",")]
PERF_TILES = int(os.environ.get("GV_TILES", "8"))
PERF_REPS = [int(v) for v in os.environ.get("GV_REPS", "0,32").split(",")]


@pytest.mark.parametrize("out", ["bf16", "fp32"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_precision(device, variant, out):
    import torch

    x = bench.sweep_input(64)
    dt = ttnn.bfloat16 if out == "bf16" else ttnn.float32
    got = bench.run(device, x, variant=variant, reps=1, out_dtype=dt)
    rep = bench.precision_report(x.to(torch.float64), got, out == "bf16")
    print(f"PRECISION variant={variant} out={out} " + " ".join(f"{k}={v:.4g}" for k, v in rep.items()))
    assert torch.isfinite(got).all()
    if out == "bf16":
        assert rep["max_ulp_masked"] <= 2, rep


@pytest.mark.parametrize("reps", PERF_REPS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_perf(device, variant, reps):
    x = bench.sweep_input(PERF_TILES)
    bench.run(device, x, variant=variant, reps=reps, out_dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)
    print(f"LAUNCH variant={variant} reps={reps} tiles={PERF_TILES}")


@pytest.mark.parametrize("variant", VARIANTS)
def test_bias_bins(device, variant):
    """fp32-output signed error binned over x (diagnostic)."""
    import torch

    x = bench.sweep_input(64).to(torch.float64)
    got = bench.run(device, x, variant=variant, reps=1, out_dtype=ttnn.float32)
    ref = bench.gelu_ref(x)
    err = (got - ref).flatten()
    xf = x.flatten()
    edges = [-12, -8, -5.5, -4, -2, -1, 0, 1, 2, 4, 6, 8, 12.01]
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (xf >= lo) & (xf < hi)
        parts.append(f"[{lo},{hi}):{err[m].mean().item():.2e}/{err[m].abs().max().item():.1e}")
    print(f"PRECISION bins variant={variant} " + " ".join(parts))
