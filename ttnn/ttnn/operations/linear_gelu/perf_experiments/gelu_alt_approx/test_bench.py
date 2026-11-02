# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""gelu_alt_approx: hardware probes, precision gate, and isolated pack-thread perf.

  scripts/run_safe_pytest.sh <dir>/test_bench.py -k probe          # approx_exp / approx_recip / rounding
  scripts/run_safe_pytest.sh <dir>/test_bench.py -k precision      # dense fp32 sweep vs torch gelu tanh
  GA_VARIANTS=base,t21 scripts/run_safe_pytest.sh --profile <dir>/test_bench.py -k perf
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import ga_bench as bench  # noqa: E402

HERE = Path(__file__).parent
NX, NY = 8, 8
TPC = 24  # tiles per core (multiple of block=3)


def _cores():
    return bench.grid(NX, NY)


def _to_dev(device, x):
    return ttnn.from_torch(
        x.reshape(-1, 32).contiguous(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.shard_cfg(_cores(), TPC),
    )


def _n():
    return NX * NY * TPC * 1024


def _sweep_input(seed=0):
    import torch

    n = _n()
    g = torch.Generator().manual_seed(seed)
    dense = torch.linspace(-12, 12, n * 2 // 3, dtype=torch.float64).float()
    normal = (torch.randn(n - dense.numel(), generator=g) * 3).float()
    return torch.cat([dense, normal])


def _run_vals(device, x, variant, out_dtype):
    out = bench.run(_to_dev(device, x), variant=variant, cores=_cores(), tiles_per_core=TPC, out_dtype=out_dtype)
    return ttnn.to_torch(out).reshape(-1).double()


def gelu_ref(x64):
    import torch

    k = np.sqrt(2 / np.pi)
    return 0.5 * x64 * (1 + torch.tanh(k * (x64 + 0.044715 * x64**3)))


def bf16_ulp(v):
    import torch

    a = v.abs().clamp_min(1e-38)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def test_probe(device):
    import torch

    n = _n()
    res = {}
    # approx_exp over [-90, 90]
    x = torch.linspace(-90, 90, n).float()
    y = _run_vals(device, x, "probe_exp", ttnn.float32)
    ref = torch.exp(x.double())
    m = (ref > 1e-37) & (ref < 1e37)
    rel = (y[m] / ref[m] - 1).abs()
    res["exp_max_rel"] = rel.max().item()
    res["exp_mean_rel"] = (y[m] / ref[m] - 1).mean().item()
    for lo, hi in [(-1, 1), (-10, 0), (0, 10)]:
        mm = m & (x >= lo) & (x <= hi)
        res[f"exp_max_rel_[{lo},{hi}]"] = (y[mm] / ref[mm] - 1).abs().max().item()
    res["exp_at_-90,-88,88,90"] = [y[(x - v).abs().argmin()].item() for v in (-90, -88, 88, 90)]
    pts = (-20, -10, -5, -3, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20)
    res["exp_table"] = {str(v): [x[(x - v).abs().argmin()].item(), y[(x - v).abs().argmin()].item()] for v in pts}
    # approx_recip over mantissas [1, 2) and wide
    x = torch.linspace(1, 2, n).float()
    y = _run_vals(device, x, "probe_recip", ttnn.float32)
    eps = 1 - y * x.double()
    res["recip_max_rel"] = eps.abs().max().item()
    res["recip_mean_eps"] = eps.mean().item()
    res["recip_mean_eps2"] = (eps**2).mean().item()
    res["recip_max_eps2"] = (eps**2).max().item()
    # rounding of (x + 1.5*2^23) - 1.5*2^23
    x = torch.linspace(-4, 4, n).float()
    y = _run_vals(device, x, "probe_round", ttnn.float32)
    f = x.double() - y
    res["round_f_min_max"] = [f.min().item(), f.max().item()]
    res["round_is_rne"] = bool(((f >= -0.5) & (f <= 0.5)).all().item())
    print(json.dumps(res, indent=1))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "probe.json").write_text(json.dumps(res, indent=1))


def precision_stats(x, y32, ybf):
    import torch

    x64 = x.double()
    ref = gelu_ref(x64)
    ref_bf = ref.float().to(torch.bfloat16).double()  # torch-exact value rounded once to bf16
    err = y32 - ref
    gated = ref.abs() >= 1e-3
    ulp = bf16_ulp(ref_bf)
    ulp_err = (ybf - ref_bf).abs() / ulp
    rel = err[gated].abs() / ref[gated].abs()
    return {
        "max_abs_err_fp32": err.abs().max().item(),
        "max_rel_err_fp32_gated": rel.max().item(),
        "mean_signed_err_fp32": err.mean().item(),
        "mean_signed_rel_err_fp32_gated": (err[gated] / ref[gated].abs()).mean().item(),
        "max_bf16_ulp_gated": ulp_err[gated].max().item(),
        "max_bf16_ulp_all": ulp_err.max().item(),
        "frac_bf16_mismatch_gated": (ulp_err[gated] > 0).double().mean().item(),
        "mean_signed_ulp_bf16_gated": ((ybf - ref_bf)[gated] / ulp[gated]).mean().item(),
        "max_abs_err_bf16_tail": (ybf - ref)[~gated].abs().max().item(),
        "mean_signed_err_bf16_tail": (ybf - ref)[~gated].mean().item(),
        "nonfinite": int((~torch.isfinite(y32)).sum().item()),
    }


def test_precision(device):
    import torch

    variants = os.environ.get("GA_VARIANTS", "base,t21,mag,hwx").split(",")
    x = _sweep_input()
    out = {}
    for v in variants:
        y32 = _run_vals(device, x, v, ttnn.float32)
        ybf = _run_vals(device, x, v, ttnn.bfloat16)
        out[v] = precision_stats(x, y32, ybf)
        print(v, json.dumps(out[v], indent=1))
    # extreme inputs: huge magnitudes, zeros
    xe = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e4, -1e4, 1e30, -1e30, 50.0, -50.0, 9.75, -9.75, 12.0, -12.0])
    xe = torch.cat([xe, torch.zeros(_n() - xe.numel())])
    for v in variants:
        ye = _run_vals(device, xe, v, ttnn.float32)[:14]
        out[v]["extremes"] = ye.tolist()
        print(v, "extremes", ye.tolist())
    (HERE / "results").mkdir(exist_ok=True)
    tag = os.environ.get("GA_TAG", "precision")
    (HERE / "results" / f"{tag}.json").write_text(json.dumps(out, indent=1))


PERF_REPS = int(os.environ.get("GA_REPS", "40"))


@pytest.mark.parametrize("variant", os.environ.get("GA_VARIANTS", "none,base,t21,mag,hwx").split(","))
def test_perf(device, variant):
    import torch

    torch.manual_seed(0)
    x = torch.randn(_n()) * 3
    xd = _to_dev(device, x)
    bench.run(xd, variant=variant, cores=_cores(), tiles_per_core=TPC, reps=PERF_REPS)
    ttnn.synchronize_device(device)
    with open(HERE / "results" / "perf_order.txt", "a") as f:
        f.write(f"{variant} reps={PERF_REPS} tiles={TPC}\n")
