# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""sfpu_exp_bf16: isolated bake-off of the softmax ScaledExp at bf16 DEST (HiFi2, approx=False).

  EXP_VARIANTS=base,trim,split,split2,fold2   (default all)   EXP_SQ / EXP_SK (default 16 / 8)
  scripts/run_safe_pytest.sh --run-all <this file> -s

Correctness (the only pass/fail): every variant vs a float64 torch exp of the DEVICE's own bf16 (S - m) (read back
from the `noexp` ablation), bf16 ULP stats; -inf lanes must be exactly 0; bit-mismatch count vs `base` reported.
Perf: DEVICE KERNEL DURATION [ns] of one fresh run per variant (measured, never asserted).
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import sys
from pathlib import Path

import importlib

torch = importlib.import_module("torch")  # bench-only file: keeps the ttnn package import torch-free
import ttnn
from loguru import logger

from ttnn.operations.ring_joint_scaled_dot_product_attention.perf_experiments.sfpu_exp_bf16 import bench  # noqa: E402

SCALE = float(os.environ.get("EXP_SCALE", 128**-0.5))
S_BF16 = torch.tensor(SCALE).to(torch.bfloat16).to(torch.float64).item()
_KEY = "DEVICE KERNEL DURATION [ns]"
ALL = ["base", "trim", "split2", "fold2", "split2c", "fold2c", "split2m", "fold2m", "base_after"]
VARIANTS = os.environ.get("EXP_VARIANTS", ",".join(ALL)).split(",")
SQ = int(os.environ.get("EXP_SQ", "16"))
SK = int(os.environ.get("EXP_SK", "8"))
UNGUARDED = {"split", "split2", "fold2"}  # known wrong on t<=0 / -inf lanes (no clamp): reported, not gated
OUT_JSON = Path(__file__).parent / os.environ.get("EXP_OUT", "results.json")


def _kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    tot = None
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                tot = (tot or 0.0) + float(e.duration)
    return tot


def _ord(b):  # bf16 bits (int32 tensor) -> monotone integer
    b = b & 0xFFFF
    return torch.where(b >= 0x8000, 0x8000 - b, b)


def _bits(t):
    return t.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF


def _ulp_stats(y, d):
    """y: device bf16 output, d: device bf16 (S - m). ref = exp(s*d) in float64, rounded to bf16 (RNE)."""
    ref = torch.exp(d.to(torch.float64) * S_BF16)
    neginf = torch.isneginf(d)
    tiny = (ref < 1.1754943508222875e-38) & ~neginf  # below bf16 min-normal: device flushes to 0
    normal = ~neginf & ~tiny
    ulp = (_ord(_bits(y)) - _ord(_bits(ref.to(torch.float32)))).abs()[normal]
    return {
        "max_ulp": int(ulp.max().item()),
        "mean_ulp": float(ulp.double().mean().item()),
        "frac_gt1ulp": float((ulp > 1).double().mean().item()),
        "neginf_exact0": bool((y[neginf] == 0).all().item()) if neginf.any() else None,
        "n_neginf": int(neginf.sum().item()),
        "tiny_all0": bool((y[tiny] == 0).all().item()) if tiny.any() else None,
        "n_tiny": int(tiny.sum().item()),
        "finite": bool(torch.isfinite(y).all().item()),
    }


def _scores(kind, gen):
    H, W = SQ * 32, SK * 32
    if kind == "normal":
        S = torch.randn(H, W, generator=gen) * 3.0
    elif kind == "large":
        S = torch.randn(H, W, generator=gen) * 36.0
    elif kind == "pos":  # x*s spans [0, 88] (outside softmax's domain; documents the candidate's upper range)
        S = torch.rand(H, W, generator=gen) * (88.0 / SCALE)
    else:  # sweep: x*s spans [-97, 0] uniformly, exact 0 at each row max
        S = -torch.rand(H, W, generator=gen) * (97.0 / SCALE)
        S[:, 0] = 0.0
    S = S.to(torch.bfloat16).float()
    if kind == "pos":  # m = 0 so x = S >= 0, no mask
        return S.to(torch.bfloat16), torch.zeros(H, 32).to(torch.bfloat16)
    S[:, W - 20 :] = float("-inf")  # key-tail mask
    S[torch.rand(H, W, generator=gen) < 0.01] = float("-inf")
    S[:, 0] = torch.where(torch.isinf(S[:, 0]), torch.zeros_like(S[:, 0]), S[:, 0])  # keep >=1 finite per row
    m = S.max(dim=1, keepdim=True).values
    M = m.expand(H, 32).contiguous()
    return S.to(torch.bfloat16), M.to(torch.bfloat16)


def _rescale_inputs(kind, gen):
    H = SQ * 32
    std = 36.0 if kind == "large" else 5.0
    acc = (torch.randn(H, 32, generator=gen) * std).to(torch.bfloat16).float()
    blk = acc + (torch.randn(H, 32, generator=gen) * std * 0.3).abs()
    eq = torch.rand(H, 32, generator=gen) < 0.2
    blk[eq] = acc[eq]  # equal maxima -> exp(0) = 1
    blk = torch.maximum(blk.to(torch.bfloat16).float(), acc)
    return acc.to(torch.bfloat16), blk.to(torch.bfloat16)


def _run_block(device, variant, S, M):
    s_dev, m_dev = bench.to_dev(S, device), bench.to_dev(M, device)
    _kernel_ns(device)  # flush
    bench.run_exp_block(s_dev, m_dev, variant=variant, Sq=SQ, Sk=SK, scale=SCALE)
    ttnn.synchronize_device(device)
    ns = _kernel_ns(device)
    return ttnn.to_torch(s_dev).float(), ns


def _run_rescale(device, variant, A, B):
    a_dev, b_dev = bench.to_dev(A, device), bench.to_dev(B, device)
    o_dev = bench.to_dev(torch.zeros_like(A), device)
    _kernel_ns(device)
    bench.run_rescale(a_dev, b_dev, o_dev, variant=variant, Sq=SQ, scale=SCALE)
    ttnn.synchronize_device(device)
    ns = _kernel_ns(device)
    return ttnn.to_torch(o_dev).float(), ns


def test_sfpu_exp_bf16(device):
    gen = torch.Generator().manual_seed(1234)
    results, failures = {}, []
    BLOCK_KINDS = tuple(os.environ.get("EXP_BLOCK_KINDS", "normal,large,sweep").split(","))
    RESCALE_KINDS = tuple(k for k in os.environ.get("EXP_RESCALE_KINDS", "normal,large").split(",") if k)
    for phase, kinds in (("block", BLOCK_KINDS), ("rescale", RESCALE_KINDS)):
        for kind in kinds:
            if phase == "block":
                A, B = _scores(kind, gen)
                d, ns_floor = _run_block(device, "noexp", A, B)
                runner = lambda v: _run_block(device, v, A, B)
            else:
                A, B = _rescale_inputs(kind, gen)
                d, ns_floor = _run_rescale(device, "noexp", A, B)
                runner = lambda v: _run_rescale(device, v, A, B)
            key = f"{phase}/{kind}"
            results[key] = {"noexp_ns": ns_floor}
            logger.info(f"{key:16s} noexp   ns={ns_floor}")
            ybase = None
            for v in VARIANTS:
                y, ns = runner(v)
                st = _ulp_stats(y, d)
                st["ns"] = ns
                if v == "base":
                    ybase = y
                if ybase is not None:
                    st["bit_mismatch_vs_base"] = int((_bits(y) != _bits(ybase)).sum().item())
                results[key][v] = st
                logger.info(f"{key:16s} {v:7s} ns={ns} {st}")
                bad = (not st["finite"]) or st["neginf_exact0"] is False or st["max_ulp"] > 8
                if bad and v not in UNGUARDED:
                    failures.append(f"{key}/{v}: {st}")
    OUT_JSON.write_text(json.dumps(results, indent=1))
    logger.info(f"wrote {OUT_JSON}")
    assert not failures, "\n".join(failures)
