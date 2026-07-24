# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the rms_norm PASS 2 batch-x-fuse micro-bench.

Isolates the cross-core PASS 2 (x*rstd*gamma) on ONE core, everything resident in L1 (rstd
pre-supplied — no pass 1, no cross-core round). The 4-way menu reveals whether BATCH (across the C
tile-rows of a round) and FUSE (eliminate the cb_norm round-trip via an FPU DEST-reuse ·gamma against
a pre-replicated gamma) COMPOSE, or one supersedes the other:

    baseline    per-tile-row 2-chain through cb_norm            (the graduated Perf-1 pass2)
    batch_only  ONE 2-chain 2D-grid walk across C rows          (lever A alone)
    fuse_only   per-tile-row ONE fused chain, no cb_norm         (lever B alone)
    batch_fuse  ONE fused 2D-grid chain across C rows, no cb_norm (A+B composed)

Every variant computes IDENTICAL math at IDENTICAL precision (the x*rstd intermediate is bf16 in BOTH
paths — cb_norm is bf16 and DEST with fp32_dest_acc_en=False is bf16 — so the fusion is
precision-neutral). Correctness (PCC vs an fp32 torch reference) is the only pass/fail; perf is
measured (DEVICE KERNEL DURATION [ns] via ReadDeviceProfiler) and reported, never asserted.

    scripts/run_safe_pytest.sh --profile \
      ttnn/ttnn/operations/rms_norm/perf_experiments/pass2_batch_and_fuse/test_pass2_batch_and_fuse.py
"""

from __future__ import annotations

import os

# Device-profiler env — MUST be set before ttnn opens the device (perf-measure discipline).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics

import pytest

import ttnn
from loguru import logger

from ttnn.operations.rms_norm.perf_experiments.pass2_batch_and_fuse import (
    VARIANTS,
    BASELINE,
    variant_is_valid,
    cb_norm_depth_for,
    create_sharded_memory_config,
    run_op,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# The perf-flagged focus geometry: BLOCK_SHARDED (1,1,8192,1024) 8x8 critical core, pass 2.
FOCUS = dict(per_w_t=4, ht_local=32, c_rows=8)

SOFT_PCC_GATE = 0.9995  # below this a variant is disqualified (reported with its precision cost)
CORRECTNESS_FLOOR = 0.99  # gross-bug catch for the pass/fail assert; real gate is SOFT_PCC_GATE


# =============================================================================
# Inputs + torch golden for x*rstd*gamma of a resident tile block
# =============================================================================
def _make_case(device, per_w_t, ht_local, has_gamma=True, seed=7):
    """Resident x (bf16), rstd (fp32, per-row Col-broadcast), gamma (bf16, PRE-REPLICATED to [32,W]).

    rstd is a REDUCE_ROW-shaped column result: a per-tile-row scalar replicated across the tile's 32
    columns (so the Col broadcast is exact). gamma is PRE-REPLICATED down all 32 rows so the fused
    NO-broadcast mul reads the correct per-column value on every row (this is lever B's data prep —
    the same storage as a [1,W] gamma tile, just every row filled). The unfused Row-broadcast reads
    only row 0, giving the SAME result — so both paths see identical gamma data.
    """
    import torch

    torch.manual_seed(seed)
    m = ht_local * TILE
    n = per_w_t * TILE

    x = (torch.rand(m, n) * 2 - 1).to(torch.bfloat16).to(torch.float32)  # [-1, 1] bf16-quantized
    rstd_col = (torch.rand(m) * 1.5 + 0.25).to(torch.float32)  # [0.25, 1.75], one per global row
    gamma_row = (torch.rand(n) * 2 - 1).to(torch.bfloat16).to(torch.float32) if has_gamma else torch.ones(n)

    rstd_full = rstd_col[:, None].expand(m, TILE).contiguous().to(torch.float32)  # [m, 32]
    gamma_full = gamma_row[None, :].expand(TILE, n).contiguous().to(torch.float32)  # [32, n] replicated

    expected = x * rstd_col[:, None] * gamma_row[None, :]  # [m, n], fp32 reference

    x_dev = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((m, n)),
    )
    rstd_dev = ttnn.from_torch(
        rstd_full,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((m, TILE)),
    )
    gamma_dev = None
    if has_gamma:
        gamma_dev = ttnn.from_torch(
            gamma_full.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=create_sharded_memory_config((TILE, n)),
        )
    return x_dev, rstd_dev, gamma_dev, expected


def _pcc(actual, expected):
    import torch

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    if torch.allclose(a, e):
        return 1.0
    am, em = a - a.mean(), e - e.mean()
    denom = torch.sqrt((am * am).sum()) * torch.sqrt((em * em).sum())
    if denom == 0:
        return 0.0
    return float((am * em).sum() / denom)


def _check(output, expected, label, min_pcc=CORRECTNESS_FLOOR):
    import torch

    actual = ttnn.to_torch(output).to(torch.float32)
    pcc = _pcc(actual, expected)
    assert pcc >= min_pcc, f"{label}: PCC {pcc:.5f} < {min_pcc} (gross bug)"
    return pcc


# =============================================================================
# In-process device-kernel timing (validated round-1 pattern — do not reinvent)
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warm-up window
    samples = {name: [] for name in runners}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {name}"
            if trial:  # discard first timed pass (pipeline fill)
                samples[name].append(duration / kernel_iters)
    return samples


def _int(name, default):
    return int(os.environ.get(name, default))


def _arch_label(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    try:
        yield dev
    finally:
        ttnn.close_device(dev)


# =============================================================================
# Correctness — every variant must match torch on the focus geometry + a few more
# =============================================================================
# The batched variants walk ONE grid(C, PER_W_T) chain across the round's tile-rows, so a partial C
# block (a short last round, C∤HT_LOCAL) is not expressible — they are gated to C | HT_LOCAL. The
# per-row variants (baseline, fuse_only) loop cc over C_this and handle short rounds trivially. Every
# geometry MEASURED in this bench (focus ht=32/C=8; sweep ht=16/C∈{4,8}) divides evenly, so the gate
# never bites the reported numbers; it is the predicate boundary the coordinator would guard on.
_BATCHED = {"batch_only", "batch_fuse"}


def _applicable(variant, ht_local, c_rows):
    if variant in _BATCHED and ht_local % c_rows != 0:
        return False
    return True


def test_pass2_correctness(device):
    cases = [
        dict(per_w_t=4, ht_local=32, c_rows=8),  # focus
        dict(per_w_t=4, ht_local=32, c_rows=4),
        dict(per_w_t=2, ht_local=16, c_rows=8),
        dict(per_w_t=8, ht_local=16, c_rows=4),
        dict(per_w_t=4, ht_local=30, c_rows=8),  # short last round (30 % 8 != 0): per-row variants only
    ]
    for case in cases:
        x, rstd, gamma, expected = _make_case(device, case["per_w_t"], case["ht_local"])
        for variant in VARIANTS:
            if not variant_is_valid(variant, case["c_rows"], case["c_rows"], case["ht_local"]):
                continue
            if not _applicable(variant, case["ht_local"], case["c_rows"]):
                logger.info(f"{variant:12s} {case}  SKIP (C∤HT: batched grid needs C | HT_LOCAL)")
                continue
            out = run_op(x, rstd, gamma, variant=variant, has_gamma=True, kernel_iters=2, **case)
            pcc = _check(out, expected, f"{variant} {case}")
            ttnn.deallocate(out)
            logger.info(f"{variant:12s} {case}  PCC={pcc:.5f}")
        for t in (x, rstd, gamma):
            if t is not None:
                ttnn.deallocate(t)


# =============================================================================
# Device perf — the 4-way menu, focus geometry + predicate sweep
# =============================================================================
def _perf_one_geometry(device, per_w_t, ht_local, c_rows, trials, kernel_iters):
    x, rstd, gamma, expected = _make_case(device, per_w_t, ht_local)

    pccs = {}
    for variant in VARIANTS:
        out = run_op(
            x,
            rstd,
            gamma,
            variant=variant,
            per_w_t=per_w_t,
            ht_local=ht_local,
            c_rows=c_rows,
            has_gamma=True,
            kernel_iters=1,
        )
        pccs[variant] = _check(out, expected, f"{variant} pwt={per_w_t} ht={ht_local} c={c_rows}")
        ttnn.deallocate(out)

    runners = {
        variant: (
            lambda v=variant: run_op(
                x,
                rstd,
                gamma,
                variant=v,
                per_w_t=per_w_t,
                ht_local=ht_local,
                c_rows=c_rows,
                has_gamma=True,
                kernel_iters=kernel_iters,
            )
        )
        for variant in VARIANTS
    }
    samples = _measure(device, runners, trials, kernel_iters)
    for t in (x, rstd, gamma):
        if t is not None:
            ttnn.deallocate(t)
    ttnn.synchronize_device(device)
    return samples, pccs


def _fmt_row(variant, samples, pccs, base_med, per_w_t, c_rows):
    med = statistics.median(samples[variant])
    std = statistics.pstdev(samples[variant]) if len(samples[variant]) > 1 else 0.0
    speedup = f"{base_med / med:.3f}x" if base_med else "-"
    gate = "" if pccs.get(variant, 0.0) >= SOFT_PCC_GATE else " (BELOW 0.9995)"
    return (
        f"| {variant} | {cb_norm_depth_for(variant, per_w_t, c_rows)} | {med:.1f} | "
        f"{std / med * 100:.1f}% | {speedup} | {pccs.get(variant, float('nan')):.5f}{gate} |"
    )


def test_pass2_device_perf_focus(device):
    trials = _int("P2_TRIALS", "7")
    kernel_iters = _int("P2_KERNEL_ITERS", "50")
    per_w_t, ht_local, c_rows = FOCUS["per_w_t"], FOCUS["ht_local"], FOCUS["c_rows"]

    samples, pccs = _perf_one_geometry(device, per_w_t, ht_local, c_rows, trials, kernel_iters)
    base_med = statistics.median(samples[BASELINE])

    lines = [
        "# rms_norm PASS 2 batch-x-fuse — FOCUS geometry (single core, resident L1)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  N={trials} (median)  "
        f"kernel-iters={kernel_iters} (steady-state)",
        f"geometry: PER_W_T={per_w_t} HT_LOCAL={ht_local} C_ROWS={c_rows} "
        f"num_rounds={(ht_local + c_rows - 1)//c_rows}  dtype=bf16(x,gamma)/fp32(rstd)  "
        f"HiFi2 fp32_dest_acc=False (FROZEN)",
        "",
        "Metric: DEVICE KERNEL DURATION [ns] per full pass-2 (all HT_LOCAL tile-rows). "
        f"Speedup = {BASELINE} / variant. Correctness gate: PCC >= {SOFT_PCC_GATE}.",
        "",
        "| Variant | cb_norm tiles | Median ns | Std/med | Speedup | PCC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        lines.append(_fmt_row(variant, samples, pccs, base_med, per_w_t, c_rows))
    logger.info("\n" + "\n".join(lines) + "\n")


def test_pass2_predicate_sweep(device):
    trials = _int("P2_TRIALS", "5")
    kernel_iters = _int("P2_KERNEL_ITERS", "50")

    # Predicate axes (per the task): PER_W_T / vwt in {2,4,8} x C in {4,8}. HT_LOCAL=16 keeps the
    # per-geometry resident-shard footprint in L1 across the sweep (batching amortization depends on
    # chains/round = f(C, PER_W_T), not on HT_LOCAL / num_rounds).
    sweep = []
    for pwt in (2, 4, 8):
        for c in (4, 8):
            sweep.append(dict(per_w_t=pwt, ht_local=16, c_rows=c))

    lines = [
        "# rms_norm PASS 2 batch-x-fuse — predicate sweep (single core, resident L1)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  N={trials} (median)  kernel-iters={kernel_iters}",
        "",
        "| PER_W_T | HT | C | rounds | baseline ns | batch_only ns | fuse_only ns | batch_fuse ns | "
        "best speedup | best variant | PCC(bf) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for case in sweep:
        pwt, ht, c = case["per_w_t"], case["ht_local"], case["c_rows"]
        samples, pccs = _perf_one_geometry(device, pwt, ht, c, trials, kernel_iters)
        base = statistics.median(samples[BASELINE])
        meds = {v: statistics.median(samples[v]) for v in VARIANTS}
        best_v = min(meds, key=meds.get)
        rounds = (ht + c - 1) // c
        lines.append(
            f"| {pwt} | {ht} | {c} | {rounds} | {base:.0f} | {meds['batch_only']:.0f} | "
            f"{meds['fuse_only']:.0f} | {meds['batch_fuse']:.0f} | {base / meds[best_v]:.3f}x | {best_v} | "
            f"{pccs.get('batch_fuse', float('nan')):.5f} |"
        )
    logger.info("\n" + "\n".join(lines) + "\n")
