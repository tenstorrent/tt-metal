# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device bench for the isolated rms_norm PASS 2 (x*rstd*gamma) cb_norm-elimination idea.

Correctness is the only pass/fail: every variant is PCC-gated against the fp32 torch reference of
out = x * rstd * gamma (Col-broadcast rstd, Row-broadcast gamma). Perf is measured (median of N
fresh trials of the in-kernel steady-state loop) and reported, never asserted. The focus-case soft
PCC gate is 0.9995 — every variant's PCC is reported against it (below = flagged, precision cost).

    scripts/run_safe_pytest.sh --profile \\
        ttnn/ttnn/operations/rms_norm/perf_experiments/pass2_fuse_skip_norm/test_pass2_fuse_skip_norm.py
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

from .program_descriptor_with_inline_kernels import (
    VARIANTS,
    BASELINE,
    create_sharded_memory_config,
    run_pass2,
)

TILE = 32
_PROFILER_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Focus case: block-sharded (1,1,8192,1024) on an 8x8 grid -> per-core PER_W_T=4, HT_LOCAL=32, C=8.
# Predicate sweep: per_w_t/vwt in {1,2,4,8} (tile-aligned, no pad tail) x C_BLOCK in {1,4,8}.
# (per_w_t, ht_local, c_block).  pw8 trims ht_local to 16 to keep single-core L1 bounded.
SWEEP = [
    ("focus_pw4_C8", dict(per_w_t=4, ht_local=32, c_block=8)),
    ("pw1_C8", dict(per_w_t=1, ht_local=32, c_block=8)),
    ("pw2_C8", dict(per_w_t=2, ht_local=32, c_block=8)),
    ("pw8_C8", dict(per_w_t=8, ht_local=16, c_block=8)),
    ("pw4_C1", dict(per_w_t=4, ht_local=32, c_block=1)),
    ("pw4_C4", dict(per_w_t=4, ht_local=32, c_block=4)),
]

TRIALS = 5
KERNEL_ITERS = 40
SOFT_PCC_GATE = 0.9995  # focus-case gate; below is a precision regression (reported, never hidden)
HARD_PCC = 0.99  # correctness sanity floor (a variant below this is disqualified / broken)

_report_lines = []


def _pcc(golden, computed) -> float:
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
    # gamma: [1, W] row vector. Row 0 is the applied vector (both the baseline Row-broadcast and the
    # fused unary_bcast<ROW> replication read row 0); replicate down 32 rows for a valid TILE tensor.
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

    # fp32 torch reference: out = x * rstd_col * gamma_row.
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
            ttnn.deallocate(out)
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
        ttnn.deallocate(out)
        pcc = _pcc(ref, got)
        results[variant] = (median_ns, std_ns, pcc)

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
        if pcc != pcc or pcc <= HARD_PCC:  # NaN or below the correctness floor
            gate = "  [DISQUALIFIED: broken/NaN]"
        elif pcc < SOFT_PCC_GATE:
            gate = "  [PCC<0.9995 PRECISION COST]"
        else:
            gate = ""
        line = f"  {variant:<24} {ns:>10.1f} ns  (+/-{cv:4.1f}%)  pcc={pcc:.5f}{tag}{gate}"
        _report_lines.append(line)
        print(line)


def teardown_module(module):
    print("\n" + "=" * 96)
    print("PASS-2 FUSE (eliminate cb_norm round-trip) bench summary")
    print("=" * 96)
    for line in _report_lines:
        print(line)
