# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + single-core device profiling for the rms_norm pass-1 batch-rows experiment.

Isolates the pass-1 stage (square + REDUCE_ROW SUM * 1/W) of the cross-core rms_norm compute
kernel. The op's current approach (baseline) processes one tile-row per square-chain + reduce;
the idea batches the square (and optionally the reduce) across `block_rows` tile-rows per pass to
amortize the fixed per-helper-call overhead. Every variant does the IDENTICAL math at the IDENTICAL
precision (bf16 in, HiFi2, fp32_dest_acc_en=False, fp32 stat out) -- only the call granularity /
reconfig policy change. Correctness (PCC vs torch) is the only pass/fail; perf is measured, never
asserted.

Focus shape: BLOCK_SHARDED (1,1,8192,1024) on 8x8 grid -> per core HT_LOCAL=32 tile-rows,
PER_W_T=4 W-tiles, W(origin)=1024 (scaler 1/1024). C_ROWS=8 per round.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
import sys
from pathlib import Path

import ttnn
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pass1_bench as bench  # noqa: E402

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


# =============================================================================
# Inputs + golden. Quantize x to bf16 first so the golden reflects only op-internal (accumulation)
# error, not input rounding. Per-row partial = (sum over this core's slice cols of x^2) / origin_w.
# =============================================================================
def _quant(t):
    import torch

    return t.to(torch.bfloat16).to(torch.float32)


def _make_case(device, ht_local, per_w_t, origin_w, seed=7):
    import torch

    torch.manual_seed(seed)
    rows, cols = ht_local * TILE, per_w_t * TILE
    x = torch.randn(rows, cols)
    xq = _quant(x)
    # per-row local partial: Sigma_slice x^2 * (1/origin_w)  -> [rows]
    golden = (xq * xq).sum(dim=1) / float(origin_w)

    x_dev = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.create_sharded_memory_config((rows, cols)),
    )
    return x_dev, golden


def _pcc(actual, expected):
    import torch

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def _partials_from_output(output, ht_local):
    import torch

    # cb_stat_local: HT_LOCAL fp32 tiles, tile t holds tile-row t's 32 row-partials in COLUMN 0.
    actual = ttnn.to_torch(output).to(torch.float32)  # [ht_local*32, 32]
    return actual[:, 0].contiguous()  # column 0 == the per-row reduced partial


def _check(output, golden, ht_local, label, min_pcc):
    partials = _partials_from_output(output, ht_local)
    pcc = _pcc(partials, golden)
    assert pcc >= min_pcc, f"{label}: PCC {pcc:.6f} < {min_pcc}"
    return pcc


# =============================================================================
# In-process device-kernel timing (validated pattern from examples/compute_block_size).
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
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


# =============================================================================
# Variant menu. name -> (block_rows, reduce_mode, reconfig). block_rows=1 is the op baseline.
# =============================================================================
def _menu(ht_local):
    m = {
        "baseline_br1": (1, bench.REDUCE_BLOCKED, True),  # op's per-tile-row square+reduce
        "batch_sq_br2": (2, bench.REDUCE_BLOCKED, True),
        "batch_sq_br4": (4, bench.REDUCE_BLOCKED, True),
        "batch_sq_br8": (8, bench.REDUCE_BLOCKED, True),  # one full round's C=8 tile-rows
        "batch_sqonly_br8": (8, bench.REDUCE_PER_ROW, True),  # batch square, per-row reduce
        "batch_br8_noreconfig": (8, bench.REDUCE_BLOCKED, False),  # + reconfig-skip lever
    }
    return {k: v for k, v in m.items() if bench.valid_block_rows(v[0], ht_local)}


BASELINE = "baseline_br1"


def _run(x_dev, ht_local, per_w_t, origin_w, block_rows, reduce_mode, reconfig, kernel_iters):
    return bench.run_op(
        x_dev,
        ht_local=ht_local,
        per_w_t=per_w_t,
        block_rows=block_rows,
        reduce_mode=reduce_mode,
        reconfig=reconfig,
        kernel_iters=kernel_iters,
        origin_w=origin_w,
    )


# =============================================================================
# Tests
# =============================================================================
def test_pass1_correctness(device):
    ht_local = _int("P1_HT_LOCAL", "32")
    per_w_t = _int("P1_PER_W_T", "4")
    origin_w = _int("P1_ORIGIN_W", "1024")
    min_pcc = float(os.environ.get("P1_MIN_PCC", "0.99"))

    x_dev, golden = _make_case(device, ht_local, per_w_t, origin_w)
    for name, (br, rm, rc) in _menu(ht_local).items():
        out = _run(x_dev, ht_local, per_w_t, origin_w, br, rm, rc, kernel_iters=2)
        pcc = _check(out, golden, ht_local, f"{name} (br={br}, rm={rm}, reconfig={rc})", min_pcc)
        logger.info(f"{name:22s} block_rows={br} reduce_mode={rm} reconfig={rc}  PCC={pcc:.6f}")


def test_pass1_device_perf(device):
    ht_local = _int("P1_HT_LOCAL", "32")
    per_w_t = _int("P1_PER_W_T", "4")
    origin_w = _int("P1_ORIGIN_W", "1024")
    trials = _int("P1_TRIALS", "7")
    kernel_iters = _int("P1_KERNEL_ITERS", "100")
    min_pcc = float(os.environ.get("P1_MIN_PCC", "0.99"))

    x_dev, golden = _make_case(device, ht_local, per_w_t, origin_w)
    menu = _menu(ht_local)

    pccs = {}
    for name, (br, rm, rc) in menu.items():
        out = _run(x_dev, ht_local, per_w_t, origin_w, br, rm, rc, kernel_iters=1)
        pccs[name] = _check(out, golden, ht_local, f"{name}", min_pcc)

    runners = {
        name: (lambda br=br, rm=rm, rc=rc: _run(x_dev, ht_local, per_w_t, origin_w, br, rm, rc, kernel_iters))
        for name, (br, rm, rc) in menu.items()
    }
    samples = _measure(device, runners, trials, kernel_iters)

    report = _format_report(
        samples,
        pccs,
        menu,
        ht_local,
        per_w_t,
        origin_w,
        box=socket.gethostname(),
        arch=_arch_label(device),
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("P1_REPORT"):
        Path(report_path).write_text(report)


def test_pass1_predicate_sweep(device):
    """Characterize the predicate: how the win moves with the per-block payload (PER_W_T) at the
    full-round block (block_rows=8). compute_block_size lesson: the amortization win SHRINKS as the
    per-block payload grows (each call already does more real work)."""
    ht_local = _int("P1_HT_LOCAL", "32")
    origin_w = _int("P1_ORIGIN_W", "1024")
    trials = _int("P1_TRIALS", "7")
    kernel_iters = _int("P1_KERNEL_ITERS", "100")
    min_pcc = float(os.environ.get("P1_MIN_PCC", "0.99"))
    per_w_ts = [int(v) for v in os.environ.get("P1_SWEEP_PER_W_T", "2,4,8").split(",")]

    rows = []
    for per_w_t in per_w_ts:
        x_dev, golden = _make_case(device, ht_local, per_w_t, origin_w)
        variants = {
            "baseline_br1": (1, bench.REDUCE_BLOCKED, True),
            "batch_br8": (8, bench.REDUCE_BLOCKED, True),
        }
        variants = {k: v for k, v in variants.items() if bench.valid_block_rows(v[0], ht_local)}
        pccs = {}
        for name, (br, rm, rc) in variants.items():
            out = _run(x_dev, ht_local, per_w_t, origin_w, br, rm, rc, kernel_iters=1)
            pccs[name] = _check(out, golden, ht_local, f"pwt={per_w_t} {name}", min_pcc)
        runners = {
            name: (lambda br=br, rm=rm, rc=rc, p=per_w_t: _run(x_dev, ht_local, p, origin_w, br, rm, rc, kernel_iters))
            for name, (br, rm, rc) in variants.items()
        }
        samples = _measure(device, runners, trials, kernel_iters)
        base = statistics.median(samples["baseline_br1"])
        cand = statistics.median(samples["batch_br8"])
        rows.append((per_w_t, base, cand, base / cand, pccs["baseline_br1"], pccs["batch_br8"]))

    lines = [
        "# rms_norm pass-1 batch-rows -- predicate sweep (per-block payload PER_W_T)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters}",
        f"HT_LOCAL={ht_local}  origin_W={origin_w}  block_rows: baseline=1 vs batch=8  "
        f"dtype=bf16 in / fp32 stat  HiFi2  fp32_dest_acc_en=False",
        "",
        "| PER_W_T | cb_xsq tiles (br8) | baseline ns/iter | batch_br8 ns/iter | speedup | PCC base/cand |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for per_w_t, base, cand, sp, pb, pc in rows:
        lines.append(f"| {per_w_t} | {8 * per_w_t} | {base:.1f} | {cand:.1f} | {sp:.2f}x | {pb:.5f} / {pc:.5f} |")
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    if report_path := os.environ.get("P1_SWEEP_REPORT"):
        Path(report_path).write_text(report)


def _format_report(samples, pccs, menu, ht_local, per_w_t, origin_w, *, box, arch, trials, kernel_iters):
    import torch

    base = statistics.median(samples[BASELINE]) if BASELINE in samples else None
    lines = [
        "# rms_norm pass-1 batch-rows -- single-core isolated bench",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        f"per-core pass-1: HT_LOCAL={ht_local} tile-rows  PER_W_T={per_w_t} W-tiles  origin_W={origin_w}  "
        f"dtype=bf16 in / fp32 stat  HiFi2  fp32_dest_acc_en=False (FIXED)",
        "",
        f"Metric: DEVICE KERNEL DURATION [ns] per iter (= one per-core pass-1 over {ht_local} tile-rows). "
        f"Speedup = {BASELINE} / variant. Correctness gate: PCC of per-row Sigma x^2*(1/W) vs torch.",
        "",
        "| Variant | block_rows | reduce | reconfig | cb_xsq tiles | Median ns | Std/med | Speedup | PCC |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    rm_name = {bench.REDUCE_PER_ROW: "per_row", bench.REDUCE_BLOCKED: "blocked"}
    for name, (br, rm, rc) in menu.items():
        values = samples[name]
        median = statistics.median(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        speedup = f"{base / median:.2f}x" if base else "-"
        lines.append(
            f"| {name} | {br} | {rm_name[rm]} | {'on' if rc else 'off'} | {br * per_w_t} | "
            f"{median:.1f} | {std / median * 100:.1f}% | {speedup} | {pccs.get(name, float('nan')):.5f} |"
        )
    return "\n".join(lines) + "\n"
