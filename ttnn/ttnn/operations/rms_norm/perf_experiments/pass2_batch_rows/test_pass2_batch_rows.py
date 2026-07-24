# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the rms_norm pass-2 batching micro-bench.

Isolates the cross-core PASS 2 (x*rstd * gamma) on ONE core with everything resident in L1
(rstd pre-supplied — no pass 1, no cross-core round). Every variant computes identical math;
they differ only in the pass-2 chain STRUCTURE (per-tile-row vs batched across the C tile-rows
of a round) and in the reconfig policy (RECONFIG_SKIP on/off). Correctness (PCC vs torch) is the
only pass/fail; perf is measured (DEVICE KERNEL DURATION [ns] via ReadDeviceProfiler) and
reported, never asserted.

ROUND 2 focus: measure the C-batching (batch_both) as a COMPOSED gain ON TOP of the current
op's reconfig-skip baseline — i.e. (batch_both, skip=True) vs (baseline, skip=True). The
(*, skip=False) runs reproduce the round-1 pre-Perf-1 baseline for reconciliation.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics

import ttnn
from loguru import logger

from ttnn.operations.rms_norm.perf_experiments.pass2_batch_rows import (
    variant_is_valid,
    cb_norm_depth_for,
    create_sharded_memory_config,
    run_op,
)

TILE = 32
BF16_TILE = ttnn.tile_size(ttnn.bfloat16)  # 2048 B
FP32_TILE = ttnn.tile_size(ttnn.float32)  # 4096 B
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# The perf-flagged focus geometry (BLOCK_SHARDED (1,1,8192,1024) 8x8 critical core, pass 2):
FOCUS = dict(per_w_t=4, ht_local=32, c_rows=8)

# The measurement run set: (label, variant, reconfig_skip). The current op == baseline_skip;
# the candidate == batch_both_skip. Speedup in every table is computed vs baseline_skip.
BASELINE_LABEL = "baseline_skip"
RUNS = (
    ("baseline_skip", "baseline", True),  # CURRENT OP (Perf-1): per-row, reconfig-skip
    ("batch_gamma_skip", "batch_gamma", True),  # menu: gamma-only batched, reconfig-skip
    ("batch_both_skip", "batch_both", True),  # CANDIDATE: C-batched (2 chains/round), reconfig-skip
    ("baseline_noskip", "baseline", False),  # reconciliation: round-1 pre-Perf-1 baseline
    ("batch_both_noskip", "batch_both", False),  # reconciliation: round-1 candidate
)


# =============================================================================
# Inputs + torch golden for x*rstd*gamma of a resident tile block
# =============================================================================
def _make_case(device, per_w_t, ht_local, has_gamma=True, seed=7):
    """Build resident x (bf16), rstd (fp32, per-row Col-broadcast), gamma (bf16, Row-broadcast)."""
    import torch

    torch.manual_seed(seed)
    m = ht_local * TILE
    n = per_w_t * TILE

    x = (torch.rand(m, n) * 2 - 1).to(torch.bfloat16).to(torch.float32)  # [-1,1] bf16-quantized
    rstd_col = (torch.rand(m) * 1.5 + 0.25).to(torch.float32)  # [0.25,1.75], one per global row
    gamma_row = (torch.rand(n) * 2 - 1).to(torch.bfloat16).to(torch.float32) if has_gamma else torch.ones(n)

    # Broadcast-content tiles (HW Col/Row broadcast exact regardless of which lane it reads).
    rstd_full = rstd_col[:, None].expand(m, TILE).contiguous().to(torch.float32)  # [m,32]
    gamma_full = gamma_row[None, :].expand(TILE, n).contiguous().to(torch.float32)  # [32,n]

    # fp32 reference (compute the norm at fp32 as the PCC target).
    expected = x * rstd_col[:, None] * gamma_row[None, :]

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
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def _check(output, expected, label, min_pcc=0.9995):
    import torch

    actual = ttnn.to_torch(output).to(torch.float32)
    pcc = _pcc(actual, expected)
    assert pcc >= min_pcc, f"{label}: PCC {pcc:.5f} < {min_pcc}"
    return pcc


# =============================================================================
# In-process device-kernel timing (validated pattern — do not reinvent)
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
    _read_kernel_ns(device)  # discard warmup window
    samples = {name: [] for name in runners}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {name}"
            if trial:  # discard first timed pass
                samples[name].append(duration / kernel_iters)
    return samples


def _int(name, default):
    return int(os.environ.get(name, default))


def _arch_label(device):
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _valid_runs(per_w_t, c_rows, ht_local, labels=None):
    out = []
    for label, variant, skip in RUNS:
        if labels is not None and label not in labels:
            continue
        if variant_is_valid(variant, per_w_t, c_rows, ht_local):
            out.append((label, variant, skip))
    return out


def _run_all(device, per_w_t, ht_local, c_rows, trials, kernel_iters, labels=None):
    """PCC-gate + measure every run in the (filtered) run set for one geometry."""
    x, rstd, gamma, expected = _make_case(device, per_w_t, ht_local)
    runs = _valid_runs(per_w_t, c_rows, ht_local, labels)

    pccs = {}
    for label, variant, skip in runs:
        out = run_op(
            x,
            rstd,
            gamma,
            variant=variant,
            per_w_t=per_w_t,
            ht_local=ht_local,
            c_rows=c_rows,
            has_gamma=True,
            reconfig_skip=skip,
            kernel_iters=1,
        )
        pccs[label] = _check(out, expected, f"{label} pwt={per_w_t} ht={ht_local} c={c_rows}")

    runners = {
        label: (
            lambda v=variant, s=skip: run_op(
                x,
                rstd,
                gamma,
                variant=v,
                per_w_t=per_w_t,
                ht_local=ht_local,
                c_rows=c_rows,
                has_gamma=True,
                reconfig_skip=s,
                kernel_iters=kernel_iters,
            )
        )
        for label, variant, skip in runs
    }
    samples = _measure(device, runners, trials, kernel_iters)
    for t in (x, rstd, gamma):
        if t is not None:
            ttnn.deallocate(t)
    ttnn.synchronize_device(device)
    return samples, pccs, {label: variant for label, variant, _ in runs}


def _cb_norm_kb(variant, per_w_t, c_rows):
    return cb_norm_depth_for(variant, per_w_t, c_rows) * BF16_TILE / 1024.0


# =============================================================================
# Correctness — every variant × reconfig policy must match torch on several geometries
# =============================================================================
def test_pass2_correctness(device):
    # All geometries use HT_LOCAL % C_ROWS == 0 (exact division). This mirrors the op's host-gate
    # (C-batching is only enabled when Ht_local % C == 0) AND is required by this bench's
    # steady-state kernel_iters drain: the inter-iteration `cb_pop_front(cb_out, shard_tiles)` only
    # tiles cleanly across iterations when every round is full. A non-divisible last round
    # (e.g. ht=30,c=8) is byte-correct at kernel_iters=1 (PCC 0.99999) but drifts the cb_out
    # pointer under the multi-iter drain — a bench artifact, not an op behaviour, so it is excluded.
    cases = [
        dict(per_w_t=4, ht_local=32, c_rows=8),  # focus
        dict(per_w_t=4, ht_local=32, c_rows=16),
        dict(per_w_t=1, ht_local=8, c_rows=8),
        dict(per_w_t=8, ht_local=16, c_rows=4),
        dict(per_w_t=2, ht_local=32, c_rows=4),
        dict(per_w_t=4, ht_local=32, c_rows=32),  # single round
    ]
    for case in cases:
        x, rstd, gamma, expected = _make_case(device, case["per_w_t"], case["ht_local"])
        for label, variant, skip in _valid_runs(case["per_w_t"], case["c_rows"], case["ht_local"]):
            out = run_op(x, rstd, gamma, variant=variant, has_gamma=True, reconfig_skip=skip, kernel_iters=2, **case)
            pcc = _check(out, expected, f"{label} {case}")
            logger.info(f"{label:20s} {case}  PCC={pcc:.5f}")
        for t in (x, rstd, gamma):
            if t is not None:
                ttnn.deallocate(t)
        ttnn.synchronize_device(device)


# =============================================================================
# Device perf — candidate vs the reconfig-skip baseline, focus geometry
# =============================================================================
def test_pass2_device_perf_focus(device):
    trials = _int("P2_TRIALS", "9")
    kernel_iters = _int("P2_KERNEL_ITERS", "60")
    per_w_t, ht_local, c_rows = FOCUS["per_w_t"], FOCUS["ht_local"], FOCUS["c_rows"]

    samples, pccs, variant_of = _run_all(device, per_w_t, ht_local, c_rows, trials, kernel_iters)
    base_med = statistics.median(samples[BASELINE_LABEL])

    lines = [
        "# rms_norm PASS 2 C-batching — FOCUS geometry (single core, resident L1)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  N={trials} (median)  "
        f"kernel-iters={kernel_iters} (steady-state)",
        f"geometry: PER_W_T={per_w_t} HT_LOCAL={ht_local} C_ROWS={c_rows} "
        f"num_rounds={(ht_local + c_rows - 1)//c_rows}  dtype=bf16(x,gamma)/fp32(rstd)  "
        f"HiFi2 fp32_dest_acc=False (FIXED)",
        "",
        "Speedup = baseline_skip (CURRENT OP) / run. Correctness gate: PCC vs fp32 torch >= 0.9995.",
        "",
        "| Run | chains/round | cb_norm KB | Median ns | Std/med | Speedup | PCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in samples:
        variant = variant_of[label]
        med = statistics.median(samples[label])
        std = statistics.pstdev(samples[label]) if len(samples[label]) > 1 else 0.0
        cpr = 2 * c_rows if variant == "baseline" else (c_rows + 1 if variant == "batch_gamma" else 2)
        speedup = f"{base_med / med:.3f}x" if base_med else "-"
        lines.append(
            f"| {label} | {cpr} | {_cb_norm_kb(variant, per_w_t, c_rows):.0f} | {med:.1f} | "
            f"{std / med * 100:.1f}% | {speedup} | {pccs.get(label, float('nan')):.5f} |"
        )
    logger.info("\n" + "\n".join(lines) + "\n")


# =============================================================================
# Predicate sweep — where does the C-batching win hold (both at reconfig_skip=True)?
# =============================================================================
def test_pass2_predicate_sweep(device):
    trials = _int("P2_TRIALS", "7")
    kernel_iters = _int("P2_KERNEL_ITERS", "60")
    labels = ("baseline_skip", "batch_gamma_skip", "batch_both_skip")

    # C-batching axis: fix PER_W_T=4, HT_LOCAL=32, sweep C_ROWS (1 = per-row degenerate).
    # PER_W_T axis: fix C_ROWS=8, HT_LOCAL=16, sweep PER_W_T (tile-aligned so vwt==PER_W_T).
    sweep = [dict(per_w_t=4, ht_local=32, c_rows=c) for c in (1, 2, 4, 8, 16, 32)]
    sweep += [dict(per_w_t=pwt, ht_local=16, c_rows=8) for pwt in (1, 2, 8)]

    lines = [
        "# rms_norm PASS 2 C-batching — predicate sweep (single core, resident L1, reconfig_skip=True)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  N={trials} (median)  kernel-iters={kernel_iters}",
        "",
        "| PER_W_T | HT_LOCAL | C_ROWS | rounds | baseline ns | batch_gamma ns | batch_both ns | "
        "both speedup | Δcb_norm KB | PCC(both) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sweep:
        pwt, ht, c = case["per_w_t"], case["ht_local"], case["c_rows"]
        samples, pccs, _ = _run_all(device, pwt, ht, c, trials, kernel_iters, labels=labels)
        base = statistics.median(samples["baseline_skip"])

        def med_or(label):
            return statistics.median(samples[label]) if label in samples else None

        def fmt(label):
            m = med_or(label)
            return f"{m:.0f}" if m is not None else "-"

        both = med_or("batch_both_skip")
        speedup = f"{base / both:.3f}x" if both else "-"
        dnorm = _cb_norm_kb("batch_both", pwt, c) - _cb_norm_kb("baseline", pwt, c)
        rounds = (ht + c - 1) // c
        lines.append(
            f"| {pwt} | {ht} | {c} | {rounds} | {base:.0f} | {fmt('batch_gamma_skip')} | {fmt('batch_both_skip')} | "
            f"{speedup} | {dnorm:+.0f} | {pccs.get('batch_both_skip', float('nan')):.5f} |"
        )
    logger.info("\n" + "\n".join(lines) + "\n")
