# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device bench for the isolated rms_norm PASS 2 (x*rstd*gamma) variant menu.

Correctness is the only pass/fail: every variant is PCC-gated against the torch reference of
out = x * rstd * gamma (Col-broadcast rstd, Row-broadcast gamma). Perf is measured (median of N
fresh trials of the in-kernel steady-state loop) and reported, never asserted.

    scripts/run_safe_pytest.sh ttnn/ttnn/operations/rms_norm/perf_experiments/pass2_fuse_and_reconfig/test_pass2_fuse_and_reconfig.py
"""

from __future__ import annotations

import os

# Device-profiler env — MUST be set before ttnn opens the device (perf-lab discipline).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")  # silence the loud per-read profiler histograms

import statistics

import pytest

import ttnn

# perf_experiments/ is a NAMESPACE package (no __init__.py, mirroring ttnn/ttnn/operations/examples/)
# so the ttnn.operations auto-import crawl never descends here; pytest (--import-mode=importlib) walks
# up the __init__ chain, stops at the namespace perf_experiments, and imports this test as
# `pass2_fuse_and_reconfig.test_...`, so the relative import below resolves with ttnn already cached.
from .program_descriptor_with_inline_kernels import (
    VARIANTS,
    BASELINE,
    create_sharded_memory_config,
    run_pass2,
)

TILE = 32
_PROFILER_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Focus case: block-sharded (1,1,8192,1024) on an 8x8 grid -> per-core PER_W_T=4, HT_LOCAL=32, C=8.
FOCUS = dict(per_w_t=4, ht_local=32, c_block=8)

# Predicate sweep around the focus: vary the W-slice width (PER_W_T) and the row-block (C).
# (variant, per_w_t, ht_local, c_block)
SWEEP = [
    ("focus_pw4", dict(per_w_t=4, ht_local=32, c_block=8)),
    ("narrow_pw2", dict(per_w_t=2, ht_local=32, c_block=8)),
    # per_w_t=8 with ht_local=16: all tensors are SINGLE-CORE shards, so ht_local is trimmed to keep
    # the per-core L1 (x + out + stat + cb_norm) in budget while still isolating the wider W-slice.
    ("wide_pw8", dict(per_w_t=8, ht_local=16, c_block=8)),
    ("cblock4", dict(per_w_t=4, ht_local=32, c_block=4)),
    ("cblock16", dict(per_w_t=4, ht_local=32, c_block=16)),
    ("short_ht8", dict(per_w_t=4, ht_local=8, c_block=8)),
]

TRIALS = 7
KERNEL_ITERS = 50
NOISE_PCT = 3.0  # deltas within this are noise, not a win/regression

_report_lines = []


def _pcc(golden: torch.Tensor, computed: torch.Tensor) -> float:
    import torch

    g = golden.flatten().to(torch.float64)
    c = computed.flatten().to(torch.float64)
    if torch.allclose(g, c):
        return 1.0
    gm, cm = g - g.mean(), c - c.mean()
    denom = torch.sqrt((gm * gm).sum()) * torch.sqrt((cm * cm).sum())
    if denom == 0:
        return 0.0
    return float((gm * cm).sum() / denom)


def _make_inputs(device, *, per_w_t, ht_local):
    import torch

    h = ht_local * TILE
    w = per_w_t * TILE
    torch.manual_seed(1234)

    x_t = torch.randn(h, w, dtype=torch.float32)
    # gamma: [32, W] with the gamma vector replicated down every row (row 0 is the applied vector).
    gamma_vec = 0.5 + torch.rand(w, dtype=torch.float32)  # ~[0.5, 1.5]
    gamma_t = gamma_vec.unsqueeze(0).repeat(TILE, 1)
    # rstd: per-row positive scalar, replicated across the 32 columns of each stat tile (col 0 used).
    rstd_vec = 0.3 + torch.rand(h, dtype=torch.float32)  # ~[0.3, 1.3]
    stat_t = rstd_vec.unsqueeze(1).repeat(1, TILE)

    x = ttnn.from_torch(
        x_t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, w)),
    )
    gamma = ttnn.from_torch(
        gamma_t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((TILE, w)),
    )
    stat = ttnn.from_torch(
        stat_t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((h, TILE)),
    )

    # torch reference: out = x * rstd_col * gamma_row (all fp32), rounded to bf16 by the device.
    ref = x_t * rstd_vec.unsqueeze(1) * gamma_vec.unsqueeze(0)
    return (x, gamma, stat), ref


def _measure(device, run_fn):
    warm = run_fn()  # warm-up (JIT compile + pipeline fill) — discarded
    ttnn.deallocate(warm)
    ttnn.synchronize_device(device)
    per_iter = []
    out = None
    for _ in range(TRIALS):
        ttnn.ReadDeviceProfiler(device)  # flush prior window
        new_out = run_fn()
        ttnn.ReadDeviceProfiler(device)
        per_chip = ttnn.get_latest_programs_perf_data()
        total = 0.0
        for programs in (per_chip or {}).values():
            for program in programs:
                results = getattr(program, "program_analyses_results", None) or {}
                entry = results.get(_PROFILER_DURATION_KEY)
                if entry is not None:
                    total += float(entry.duration)
        per_iter.append(total / KERNEL_ITERS)
        if out is not None:
            ttnn.deallocate(out)  # free the prior trial's output — keep only the last for PCC
        out = new_out
    return statistics.median(per_iter), statistics.pstdev(per_iter), out


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


@pytest.mark.parametrize("case_name,params", SWEEP)
def test_pass2_sweep(device, case_name, params):
    import torch

    inputs, ref = _make_inputs(device, per_w_t=params["per_w_t"], ht_local=params["ht_local"])

    results = {}
    for variant in VARIANTS:

        def run_fn(v=variant):
            return run_pass2(inputs, variant=v, kernel_iters=KERNEL_ITERS, **params)

        median_ns, std_ns, out = _measure(device, run_fn)
        got = ttnn.to_torch(out).to(torch.float32)
        ttnn.deallocate(out)  # free promptly so cases don't accumulate L1 across the session
        pcc = _pcc(ref, got)
        results[variant] = (median_ns, std_ns, pcc)
        # Correctness is the ONLY gate.
        assert pcc > 0.99, f"{case_name}/{variant} PCC {pcc:.5f} below 0.99 — disqualified"

    for t in inputs:
        ttnn.deallocate(t)

    base_ns = results[BASELINE][0]
    header = (
        f"\n[{case_name}] per_w_t={params['per_w_t']} ht_local={params['ht_local']} "
        f"c_block={params['c_block']}  (1 core, bf16, HiFi2, fp32_dest=False)  "
        f"N={TRIALS} median, kernel_iters={KERNEL_ITERS}"
    )
    _report_lines.append(header)
    print(header)
    for variant in VARIANTS:
        ns, std, pcc = results[variant]
        cv = 100.0 * std / ns if ns else 0.0
        speedup = base_ns / ns if ns else 0.0
        tag = "" if variant == BASELINE else f" -> {speedup:.3f}x vs baseline"
        line = f"  {variant:<16} {ns:>10.1f} ns  (+/-{cv:4.1f}%)  pcc={pcc:.5f}{tag}"
        _report_lines.append(line)
        print(line)


def teardown_module(module):
    print("\n" + "=" * 88)
    print("PASS-2 FUSE + RECONFIG bench summary")
    print("=" * 88)
    for line in _report_lines:
        print(line)
