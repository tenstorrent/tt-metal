# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate + single-core device timing for the pass-1 fused-colsum bake-off (idea E2).

Correctness (the only pass/fail): every variant's [S; Q] row-0 lanes must match the baseline variant bit-for-bit
(same FPU square, same pack format, same reduce datapath) and the fp64 reference to a loose tolerance.
Perf: DEVICE KERNEL DURATION [ns] read in-process (ReadDeviceProfiler), ONE fresh run per variant per config
(device kernel time has no warm-up transient); the focus config gets 3 runs -> median for the headline number.
Reported, never asserted.

Env knobs:  PFC_VARIANTS (comma list), PFC_SWEEP=0/1 (default 1), PFC_RC_SLOPE=<num_rc> (second num_rc to derive
            ns/chunk as a slope; default 6), PFC_REPORT=<path> to also write the table.
"""

import os

# The device profiler and --dev (watcher) firmware do not fit the BRISC code region together: run the
# correctness test with PFC_NO_PROFILER=1 --dev, the perf test without --dev.
if os.environ.get("PFC_NO_PROFILER") != "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import pytest
import importlib

# perf-experiment test (lives under ttnn/): the repo forbids a module-level `import torch` inside ttnn/, so bind it
# through importlib — same object, same usage, no global Import node.
torch = importlib.import_module("torch")
import ttnn
from loguru import logger

from ttnn.operations.groupnorm_sc_N_1_HW_C.perf_experiments.pass1_fused_colsum.pass1_fused_colsum_bench import (
    EXACT_VARIANTS,
    FOCUS,
    TILE,
    VARIANTS,
    run_colsum,
    sharded_config,
    x_tile_width,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _selected_variants():
    sel = tuple(os.environ.get("PFC_VARIANTS", ",".join(VARIANTS)).split(","))
    unknown = set(sel) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown PFC_VARIANTS: {sorted(unknown)}")
    return sel


def _make_x(rows, cols, num_rc, hw_tail=0, seed=11):
    """Random bf16 x with a non-zero mean (cancellation visible); rows past HW (hw_tail) carry garbage the partial
    scaler must mask."""
    torch.manual_seed(seed)
    H = rows * num_rc * TILE
    x = (torch.randn(H, cols * TILE) * 1.0 + 1.5).to(torch.bfloat16)
    valid_rows = H if hw_tail == 0 else H - TILE + hw_tail
    if hw_tail:
        x[valid_rows:, :] = torch.full((H - valid_rows, cols * TILE), 77.0, dtype=torch.bfloat16)
    return x, valid_rows


def _reference(x, valid_rows):
    xf = x[:valid_rows].to(torch.float64)
    xsq = (x[:valid_rows].to(torch.float32) ** 2).to(torch.bfloat16).to(torch.float64)  # Float16_b xsq pages
    return torch.cat([xf.sum(dim=0), xsq.sum(dim=0)])  # [2c*32] lanes


def _to_device(x, device, variant, cols):
    W = x_tile_width(variant, cols)
    if W != x.shape[1] // TILE:  # interleaved_ub: x in the left half, zeros in the (to-be-squared) right half
        x = torch.cat([x, torch.zeros_like(x)], dim=1)
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config(x.shape[0] // TILE, W),
    )


def _lanes(out):
    """Row-0 lanes of the [S..;Q..] tile row -> [2c*32] fp32."""
    return ttnn.to_torch(out).to(torch.float32)[0, :]


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


def _run_once(device, variant, rows, cols, num_rc, hw_tail=0):
    x, valid_rows = _make_x(rows, cols, num_rc, hw_tail)
    tx = _to_device(x, device, variant, cols)
    out = run_colsum(tx, variant=variant, rows=rows, cols=cols, num_rc=num_rc, hw_tail=hw_tail)
    ns = _read_kernel_ns(device)
    lanes = _lanes(out)
    ttnn.deallocate(out)
    ttnn.deallocate(tx)
    return lanes, ns, _reference(x, valid_rows)


def _compare(lanes, base_lanes, ref, label):
    d_base = (lanes - base_lanes).abs().max().item()
    rel_ref = ((lanes.to(torch.float64) - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    assert torch.isfinite(lanes).all(), f"{label}: non-finite output"
    assert rel_ref < 2e-2, f"{label}: max rel diff vs fp64 reference {rel_ref:.3e}"
    return d_base, rel_ref


# Sweep grid: the focus geometry, the flagship's (cols=1, rows=4), the single-core pins' (cols=5) and the DEST edge
# (cols=8 -> 2c = 16 = two internal DEST chunks).
_ROWS = (1, 2, 4, 8)
_COLS = (1, 2, 4, 5, 8)


def test_pass1_fused_colsum_correctness(device):
    variants = _selected_variants()
    configs = [(2, 2, 2, 0), (2, 2, 2, 20), (4, 1, 3, 0), (1, 5, 2, 0), (2, 8, 2, 12), (8, 8, 2, 0)]
    for rows, cols, num_rc, hw_tail in configs:
        base_lanes, _, ref = _run_once(device, "baseline", rows, cols, num_rc, hw_tail)
        _compare(base_lanes, base_lanes, ref, f"baseline r{rows} c{cols} rc{num_rc} tail{hw_tail}")
        for variant in variants:
            if variant == "baseline":
                continue
            lanes, _, _ = _run_once(device, variant, rows, cols, num_rc, hw_tail)
            d_base, rel_ref = _compare(lanes, base_lanes, ref, f"{variant} r{rows} c{cols} rc{num_rc} tail{hw_tail}")
            logger.info(
                f"{variant:14s} rows={rows} cols={cols} num_rc={num_rc} hw_tail={hw_tail}: "
                f"max|d vs baseline|={d_base:.3e} max rel vs fp64={rel_ref:.3e}"
            )
            if variant in EXACT_VARIANTS:
                assert d_base == 0.0, f"{variant} r{rows} c{cols}: not bit-identical to baseline (max abs {d_base})"


def test_pass1_fused_colsum_device_perf(device):
    variants = _selected_variants()
    sweep = os.environ.get("PFC_SWEEP", "1") == "1"
    rc_slope = int(os.environ.get("PFC_RC_SLOPE", "6"))
    focus = (FOCUS["rows"], FOCUS["cols"])
    grid = [(r, c) for r in _ROWS for c in _COLS] if sweep else [focus]
    if focus not in grid:
        grid.insert(0, focus)

    rows_out = []  # (rows, cols, {variant: (ns_rc2, ns_rcS, rc_s)})
    L1_BUDGET = 1_000_000  # bytes for the resident x block; larger slope runs fall back to a shorter num_rc

    def slope_rc(variant, rows, cols):
        for rc in (rc_slope, 4, 3):
            if rows * x_tile_width(variant, cols) * rc * 2048 <= L1_BUDGET and rc > FOCUS["num_rc"]:
                return rc
        return 3

    for rows, cols in grid:
        per = {}
        base_ref = {}
        needed = sorted({FOCUS["num_rc"]} | {slope_rc(v, rows, cols) for v in variants})
        for num_rc in needed:  # baseline first: reference lanes for every num_rc any variant uses
            n_runs = 3 if ((rows, cols) == focus and num_rc == FOCUS["num_rc"]) else 1
            ns_list = []
            for _ in range(n_runs):
                lanes, ns, ref = _run_once(device, "baseline", rows, cols, num_rc)
                assert ns is not None, "no profiler data"
                ns_list.append(ns)
            base_ref[num_rc] = (lanes, ref, statistics.median(ns_list))
        for variant in variants:
            rc_s = slope_rc(variant, rows, cols)
            samples = {}
            for num_rc in (FOCUS["num_rc"], rc_s):
                if variant == "baseline":
                    samples[num_rc] = base_ref[num_rc][2]
                    continue
                n_runs = 3 if ((rows, cols) == focus and num_rc == FOCUS["num_rc"]) else 1
                ns_list = []
                for _ in range(n_runs):
                    lanes, ns, ref = _run_once(device, variant, rows, cols, num_rc)
                    assert ns is not None, "no profiler data"
                    ns_list.append(ns)
                samples[num_rc] = statistics.median(ns_list)
                d_base, rel_ref = _compare(lanes, base_ref[num_rc][0], ref, f"{variant} r{rows} c{cols} rc{num_rc}")
                if variant in EXACT_VARIANTS:
                    assert d_base == 0.0, f"{variant} r{rows} c{cols} rc{num_rc}: differs from baseline by {d_base}"
            per[variant] = (samples[FOCUS["num_rc"]], samples[rc_s], rc_s)
        rows_out.append((rows, cols, per))
        logger.info(_fmt_row(rows, cols, per, variants))

    report = _format_report(rows_out, variants, box=socket.gethostname(), arch=str(device.arch()))
    logger.info("\n" + report)
    if path := os.environ.get("PFC_REPORT"):
        Path(path).write_text(report)


def _chunk_ns(entry):
    n2, nS, rc_s = entry
    return (nS - n2) / (rc_s - FOCUS["num_rc"])


def _fmt_row(rows, cols, per, variants):
    tiles = rows * cols
    b_chunk = _chunk_ns(per["baseline"])
    cells = [f"r{rows} c{cols} ({tiles}t):"]
    for v in variants:
        ch = _chunk_ns(per[v])
        cells.append(f"{v}: T2={per[v][0]:.0f} ns/chunk={ch:.0f} ns/tile={ch / tiles:.0f} x{b_chunk / max(ch, 1):.2f}")
    return " | ".join(cells)


def _format_report(rows_out, variants, *, box, arch):
    lines = [
        "# pass1_fused_colsum -- single-core compute-only bake-off (idea E2)",
        "",
        f"box={box}  arch={arch}  core=(0,0)  x bf16 resident L1  fp32 DEST / HiFi4 / full sync  "
        "one fresh run per cell (focus T2: 3x median); ns/chunk = (T(num_rc=S) - T(num_rc=2)) / (S - 2), S = 6 "
        "(4 or 3 where the resident x block would exceed ~1 MB L1)",
        "",
        "| rows | cols | tiles | variant | T(num_rc=2) ns | S | ns/chunk | ns/tile | baseline/variant (per chunk) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rows, cols, per in rows_out:
        tiles = rows * cols
        b_chunk = _chunk_ns(per["baseline"])
        for v in variants:
            ch = _chunk_ns(per[v])
            lines.append(
                f"| {rows} | {cols} | {tiles} | {v} | {per[v][0]:.0f} | {per[v][2]} | {ch:.0f} | {ch / tiles:.0f} | "
                f"{b_chunk / max(ch, 1):.2f}x |"
            )
    return "\n".join(lines)
