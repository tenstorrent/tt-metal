# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the rms_norm pass-2 batching micro-bench.

Isolates the cross-core PASS 2 (x*rstd * gamma) on ONE core with everything resident in L1
(rstd pre-supplied — no pass 1, no cross-core round). Every variant computes identical math;
they differ only in the pass-2 chain STRUCTURE (per-tile-row vs batched across the C tile-rows
of a round). Correctness (PCC vs torch) is the only pass/fail; perf is measured (DEVICE KERNEL
DURATION [ns] via ReadDeviceProfiler) and reported, never asserted.
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
    VARIANTS,
    BASELINE,
    variant_is_valid,
    cb_norm_depth_for,
    create_sharded_memory_config,
    run_op,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# The perf-flagged focus geometry (BLOCK_SHARDED (1,1,8192,1024) 8x8 critical core, pass 2):
FOCUS = dict(per_w_t=4, ht_local=32, c_rows=8)


# =============================================================================
# Inputs + torch golden for x*rstd*gamma of a resident tile block
# =============================================================================
def _quant_bf16(t):
    import torch

    return t.to(torch.bfloat16).to(torch.float32)


def _make_case(device, per_w_t, ht_local, has_gamma=True, seed=7):
    """Build resident x (bf16), rstd (fp32, per-row Col-broadcast), gamma (bf16, Row-broadcast).

    rstd is a REDUCE_ROW-shaped column result: a per-tile-row scalar replicated across the
    tile's columns (so the Col broadcast is exact whatever column the HW reads). gamma is a
    row vector replicated across the tile's rows (so the Row broadcast is exact).
    """
    import torch

    torch.manual_seed(seed)
    m = ht_local * TILE
    n = per_w_t * TILE

    x = (torch.rand(m, n) * 2 - 1).to(torch.bfloat16).to(torch.float32)  # [-1, 1] bf16-quantized
    # rstd = 1/RMS is positive; keep it O(1) so bf16 intermediates don't over/underflow.
    rstd_col = (torch.rand(m) * 1.5 + 0.25).to(torch.float32)  # [0.25, 1.75], one per global row
    gamma_row = (torch.rand(n) * 2 - 1).to(torch.bfloat16).to(torch.float32) if has_gamma else torch.ones(n)

    # Broadcast-content tiles (so HW Col/Row broadcast is exact regardless of which lane it reads).
    rstd_full = rstd_col[:, None].expand(m, TILE).contiguous().to(torch.float32)  # [m, 32]
    gamma_full = gamma_row[None, :].expand(TILE, n).contiguous().to(torch.float32)  # [32, n]

    expected = x * rstd_col[:, None] * gamma_row[None, :]  # [m, n], fp32

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


def _check(output, expected, label, min_pcc=0.99):
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


def _valid_variants(per_w_t, c_rows, ht_local):
    return tuple(v for v in VARIANTS if variant_is_valid(v, per_w_t, c_rows, ht_local))


# =============================================================================
# Correctness — every variant must match torch on the focus geometry + a couple more
# =============================================================================
def test_pass2_correctness(device):
    cases = [
        dict(per_w_t=4, ht_local=32, c_rows=8),  # focus
        dict(per_w_t=4, ht_local=32, c_rows=16),
        dict(per_w_t=1, ht_local=8, c_rows=8),
        dict(per_w_t=8, ht_local=16, c_rows=4),
        dict(per_w_t=4, ht_local=30, c_rows=8),  # short last round (30 % 8 != 0)
    ]
    for case in cases:
        x, rstd, gamma, expected = _make_case(device, case["per_w_t"], case["ht_local"])
        for variant in _valid_variants(case["per_w_t"], case["c_rows"], case["ht_local"]):
            out = run_op(x, rstd, gamma, variant=variant, has_gamma=True, kernel_iters=2, **case)
            pcc = _check(out, expected, f"{variant} {case}")
            logger.info(f"{variant:16s} {case}  PCC={pcc:.5f}")


# =============================================================================
# Device perf — baseline vs batched, focus geometry + predicate sweep
# =============================================================================
def _perf_one_geometry(device, per_w_t, ht_local, c_rows, trials, kernel_iters):
    x, rstd, gamma, expected = _make_case(device, per_w_t, ht_local)
    variants = _valid_variants(per_w_t, c_rows, ht_local)

    pccs = {}
    for variant in variants:
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
        for variant in variants
    }
    samples = _measure(device, runners, trials, kernel_iters)
    # Free this geometry's resident L1 shards before the next geometry allocates (the sweep
    # walks many shapes; without this they accumulate and OOM L1 at the wide/tall cases).
    for t in (x, rstd, gamma):
        if t is not None:
            ttnn.deallocate(t)
    ttnn.synchronize_device(device)
    return samples, pccs


def _fmt_row(variant, samples, pccs, base_med, per_w_t, c_rows):
    med = statistics.median(samples[variant])
    std = statistics.pstdev(samples[variant]) if len(samples[variant]) > 1 else 0.0
    speedup = f"{base_med / med:.2f}x" if base_med else "-"
    return (
        f"| {variant} | {cb_norm_depth_for(variant, per_w_t, c_rows)} | {med:.1f} | "
        f"{std / med * 100:.1f}% | {speedup} | {pccs.get(variant, float('nan')):.5f} |"
    )


def test_pass2_device_perf_focus(device):
    import torch

    trials = _int("P2_TRIALS", "7")
    kernel_iters = _int("P2_KERNEL_ITERS", "50")
    per_w_t, ht_local, c_rows = FOCUS["per_w_t"], FOCUS["ht_local"], FOCUS["c_rows"]

    samples, pccs = _perf_one_geometry(device, per_w_t, ht_local, c_rows, trials, kernel_iters)
    base_med = statistics.median(samples[BASELINE])

    lines = [
        "# rms_norm PASS 2 batching — FOCUS geometry (single core, resident L1)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  N={trials} (median)  "
        f"kernel-iters={kernel_iters} (steady-state)",
        f"geometry: PER_W_T={per_w_t} HT_LOCAL={ht_local} C_ROWS={c_rows} "
        f"num_rounds={(ht_local + c_rows - 1)//c_rows}  dtype=bf16(x,gamma)/fp32(rstd)  "
        f"HiFi2 fp32_dest_acc=False (FIXED)",
        "",
        "Metric: DEVICE KERNEL DURATION [ns] per full pass-2 (all HT_LOCAL tile-rows). "
        f"Speedup = {BASELINE} / variant. Correctness gate: PCC vs torch.",
        "",
        "| Variant | cb_norm tiles | Median ns | Std/med | Speedup | PCC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in samples:
        lines.append(_fmt_row(variant, samples, pccs, base_med, per_w_t, c_rows))
    logger.info("\n" + "\n".join(lines) + "\n")


def test_pass2_predicate_sweep(device):
    trials = _int("P2_TRIALS", "5")
    kernel_iters = _int("P2_KERNEL_ITERS", "50")

    # Sweep C_ROWS at fixed PER_W_T (the batching-amortization axis), and PER_W_T at fixed C_ROWS.
    # HT_LOCAL=16 keeps the per-geometry resident-shard footprint inside L1 across the whole sweep
    # (the amortization depends on chains/round = f(C_ROWS, PER_W_T), not on HT_LOCAL / num_rounds).
    sweep = []
    for c in (1, 2, 4, 8, 16):
        sweep.append(dict(per_w_t=4, ht_local=16, c_rows=c))
    for pwt in (1, 2, 8):
        sweep.append(dict(per_w_t=pwt, ht_local=16, c_rows=8))

    lines = [
        "# rms_norm PASS 2 batching — predicate sweep (single core, resident L1)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  N={trials} (median)  " f"kernel-iters={kernel_iters}",
        "",
        "| PER_W_T | HT_LOCAL | C_ROWS | rounds | baseline ns | batch_gamma ns | batch_both ns | "
        "best speedup | PCC(both) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in sweep:
        pwt, ht, c = case["per_w_t"], case["ht_local"], case["c_rows"]
        samples, pccs = _perf_one_geometry(device, pwt, ht, c, trials, kernel_iters)
        base = statistics.median(samples[BASELINE])

        def med_or(v):
            return statistics.median(samples[v]) if v in samples else None

        def fmt(v):
            m = med_or(v)
            return f"{m:.0f}" if m is not None else "-"

        cands = [med_or(v) for v in ("batch_gamma", "batch_both") if med_or(v) is not None]
        best = min(cands) if cands else base
        rounds = (ht + c - 1) // c
        lines.append(
            f"| {pwt} | {ht} | {c} | {rounds} | {base:.0f} | {fmt('batch_gamma')} | {fmt('batch_both')} | "
            f"{base / best:.2f}x | {pccs.get('batch_both', float('nan')):.5f} |"
        )
    logger.info("\n" + "\n".join(lines) + "\n")
