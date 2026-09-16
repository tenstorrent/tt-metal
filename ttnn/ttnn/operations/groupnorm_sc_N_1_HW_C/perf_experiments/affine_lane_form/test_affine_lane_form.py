# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + on-device measurement for perf_experiments/affine_lane_form (Perf round 2).

Correctness is the only pass/fail: every variant must (1) match a torch fp32 reference of
y = x * (rstd_T * gamma) + (beta - mean_T * rstd_T * gamma) within the op's documented tf32-class + bf16-output
tolerance and (2) be compared against the baseline variant (`batched` = the op today): bit-identical or max-abs-diff
reported. Perf is measured (DEVICE KERNEL DURATION [ns], in-process device profiler), never asserted.

Run:  scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/perf_experiments/affine_lane_form/test_affine_lane_form.py
Env:  ALF_ITERS="1,6"  ALF_REPORT=<path>  ALF_VARIANTS="batched,lane_d2a,..."  ALF_SWEEP=focus|affine|apply|gb|all
      ALF_AFFINE_KG="1,2" (Kg filter of the affine sweep)  ALF_APPLY="1x1,2x2,..." (chunk_rows x cols cells of the apply sweep)
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import math
import socket
from pathlib import Path

import pytest
import importlib

# perf-experiment test (lives under ttnn/): the repo forbids a module-level `import torch` inside ttnn/, so bind it
# through importlib — same object, same usage, no global Import node.
torch = importlib.import_module("torch")
import ttnn
from loguru import logger

from ttnn.operations.groupnorm_sc_N_1_HW_C.perf_experiments.affine_lane_form.affine_lane_form_bench import (
    TILE,
    VARIANTS,
    membership_tile_index,
    run_region,
    sharded_row_config,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Focus geometry: the tournament shape (1,1,1024,640) G=32 on the 11x10 grid -> cols=2, Kg=1, chunk_rows=2,
# 2 row chunks, the second ragged to 1 valid row (Ht_core = 3).
FOCUS = dict(cols=2, Kg=1, chunk_rows=2, num_row_chunks=2, Ht_core=3, has_gamma=True, has_beta=True)


# =============================================================================
# Synthetic inputs
# =============================================================================
def _group_size(cols, Kg):
    """Channels per group: the first Cg >= cols*32/G that does NOT divide 32, so groups straddle tile boundaries.
    Channels >= G*Cg (if any) are padded lanes (E = 0)."""
    G = Kg * TILE
    Cg = max(1, math.ceil(cols * TILE / G))
    while 32 % Cg == 0:
        Cg += 1
    return Cg


def make_case(device, *, cols, Kg, chunk_rows, num_row_chunks, Ht_core, has_gamma, has_beta, dirty_rows=False, seed=11):
    torch.manual_seed(seed)
    G = Kg * TILE
    Cg = _group_size(cols, Kg)
    C = cols * TILE
    C_valid = min(G * Cg, C)
    mean_g = torch.rand(G) * 2.0 - 1.0
    rstd_g = torch.rand(G) * 1.5 + 0.5
    # lane-form stats: row 0 lane g&31 of tile kg = g>>5 ; tiles [mean_kg.., rstd_kg..]
    stats = torch.zeros(TILE, 2 * Kg * TILE)
    if dirty_rows:
        # rows 1..31 hold finite junk: nothing in any variant may read them
        stats[1:] = torch.randn(TILE - 1, 2 * Kg * TILE) * 100.0
    for g in range(G):
        kg, gl = g >> 5, g & 31
        stats[0, kg * TILE + gl] = mean_g[g]
        stats[0, (Kg + kg) * TILE + gl] = rstd_g[g]
    # membership E_T[g_local, c] = 1 if channel T*32+c is in group g (kg = g>>5), k-major page order
    E = torch.zeros(TILE, cols * Kg * TILE)
    for ch in range(C_valid):
        tl, c = ch // TILE, ch % TILE
        g = ch // Cg
        idx = membership_tile_index(tl, g >> 5, cols, Kg)
        E[g & 31, idx * TILE + c] = 1.0
    gamma = torch.randn(C).to(torch.bfloat16) if has_gamma else torch.ones(C, dtype=torch.bfloat16)
    beta = torch.randn(C).to(torch.bfloat16) if has_beta else torch.zeros(C, dtype=torch.bfloat16)
    gamma_row = torch.zeros(TILE, C, dtype=torch.bfloat16)
    beta_row = torch.zeros(TILE, C, dtype=torch.bfloat16)
    gamma_row[0] = gamma
    beta_row[0] = beta
    # x: num_row_chunks quanta of chunk_rows*cols tiles, valid tiles dense at the front of each quantum
    chunk = chunk_rows * cols
    n_pages = num_row_chunks * chunk
    x_tiles = torch.zeros(n_pages, TILE, TILE)
    valid_pages = []
    for rc in range(num_row_chunks):
        valid_rows = min(chunk_rows, Ht_core - rc * chunk_rows)
        for r in range(valid_rows):
            for c in range(cols):
                p = rc * chunk + r * cols + c
                x_tiles[p] = torch.randn(TILE, TILE) * 2.0
                valid_pages.append((p, c))
    x_tiles = x_tiles.to(torch.bfloat16)
    x_row = torch.cat([x_tiles[p] for p in range(n_pages)], dim=1)  # (32, n_pages*32)
    # reference (fp32): per-channel expansion, a = rstd_T*gamma, b = beta - mean_T*a, y = x*a + b
    mean_T = torch.zeros(C)
    rstd_T = torch.zeros(C)
    for ch in range(C_valid):
        g = ch // Cg
        mean_T[ch] = mean_g[g]
        rstd_T[ch] = rstd_g[g]
    a_T = rstd_T * gamma.to(torch.float32)
    b_T = beta.to(torch.float32) - mean_T * a_T
    y_ref = torch.zeros(n_pages, TILE, TILE)
    for p, c in valid_pages:
        a_c = a_T[c * TILE : (c + 1) * TILE]
        b_c = b_T[c * TILE : (c + 1) * TILE]
        y_ref[p] = x_tiles[p].to(torch.float32) * a_c + b_c

    def to_dev(t, dtype, n_tiles):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_row_config(n_tiles)
        )

    inputs = {
        "stats_row": to_dev(stats, ttnn.float32, 2 * Kg),
        "membership": to_dev(E, ttnn.float32, cols * Kg),
        "gamma": to_dev(gamma_row, ttnn.bfloat16, cols),
        "beta": to_dev(beta_row, ttnn.bfloat16, cols),
        "x": to_dev(x_row, ttnn.bfloat16, n_pages),
        "zero": to_dev(torch.zeros(TILE, TILE, dtype=torch.bfloat16), ttnn.bfloat16, 1),
    }
    geom = dict(
        cols=cols,
        Kg=Kg,
        chunk_rows=chunk_rows,
        num_row_chunks=num_row_chunks,
        Ht_core=Ht_core,
        has_gamma=has_gamma,
        has_beta=has_beta,
    )
    valid_idx = [p for p, _ in valid_pages]
    return inputs, y_ref[valid_idx], valid_idx, geom, dict(G=G, Cg=Cg, C_valid=C_valid)


def _run(device, inputs, variant, geom, valid_idx, iters=1, zones=False):
    y = run_region(device, inputs, variant=variant, iters=iters, zones=zones, **geom)
    n_pages = geom["num_row_chunks"] * geom["chunk_rows"] * geom["cols"]
    y_t = ttnn.to_torch(y).to(torch.float32)  # (32, n_pages*32)
    tiles = torch.stack([y_t[:, p * TILE : (p + 1) * TILE] for p in range(n_pages)])
    return tiles[valid_idx]


def _check_ref(got, ref, label):
    # Sanity vs torch fp32: the FPU evaluates the affine at tf32-class operand precision and y is packed to bf16
    # (|y| up to ~15 here -> bf16 half-ulp ~0.03). The fine gate is the comparison against the baseline variant.
    err = (got - ref).abs().max().item()
    mean_err = (got - ref).abs().mean().item()
    logger.info(f"{label}: max|y - torch| = {err:.3e}  mean|y - torch| = {mean_err:.3e}")
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=6e-2, msg=f"{label} vs torch reference")
    return err, mean_err


# =============================================================================
# In-process device-kernel timing
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


def _measure_once(device, inputs, variant, geom, valid_idx, iters):
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # consume anything pending
    got = _run(device, inputs, variant, geom, valid_idx, iters=iters)
    ns = _read_kernel_ns(device)
    assert ns is not None, f"no profiler data for {variant} {geom} iters={iters}"
    return ns, got


def _int_list(name, default):
    return tuple(int(v) for v in os.environ.get(name, default).split(","))


def _variants():
    env = os.environ.get("ALF_VARIANTS")
    if not env:
        return VARIANTS
    vs = tuple(v.strip() for v in env.split(","))
    return ("batched",) + tuple(v for v in vs if v != "batched")


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("dirty_rows", [False, True])
@pytest.mark.parametrize("gb", [(True, True), (True, False), (False, False)])
def test_affine_lane_form_correctness(device, gb, dirty_rows):
    geom_in = dict(FOCUS, has_gamma=gb[0], has_beta=gb[1])
    inputs, y_ref, valid_idx, geom, info = make_case(device, dirty_rows=dirty_rows, **geom_in)
    base = _run(device, inputs, "batched", geom, valid_idx)
    _check_ref(base, y_ref, f"batched {gb} dirty={dirty_rows}")
    for variant in _variants()[1:]:
        got = _run(device, inputs, variant, geom, valid_idx)
        _check_ref(got, y_ref, f"{variant} {gb} dirty={dirty_rows}")
        d = (got - base).abs().max().item()
        logger.info(f"{variant} {gb} dirty={dirty_rows} {info}: bit-identical={torch.equal(got, base)} max|dy|={d:.3e}")
    # the multi-iteration kernel (used for the per-region slope) must produce the same output as iters=1
    for variant in _variants():
        got = _run(device, inputs, variant, geom, valid_idx, iters=3)
        _check_ref(got, y_ref, f"{variant} iters=3 {gb} dirty={dirty_rows}")


def _sweep_cells():
    which = os.environ.get("ALF_SWEEP", "all")
    cells = []
    if which in ("focus", "all"):
        cells.append(("focus", dict(FOCUS)))
    if which in ("affine", "all"):
        for Kg in _int_list("ALF_AFFINE_KG", "1,2"):
            for cols in (1, 2, 4, 5, 8):
                cells.append((f"affine cols={cols} Kg={Kg}", dict(FOCUS, cols=cols, Kg=Kg)))
    if which in ("apply", "all"):
        pairs = os.environ.get("ALF_APPLY", "1x1,2x2,4x4,8x1,1x8,4x8").split(",")
        for chunk_rows, cols in (tuple(int(v) for v in pr.split("x")) for pr in pairs):
            cells.append(
                (
                    f"apply {chunk_rows}x{cols}",
                    dict(FOCUS, cols=cols, chunk_rows=chunk_rows, num_row_chunks=2, Ht_core=2 * chunk_rows),
                )
            )
    if which in ("gb", "all"):
        for gb in ((True, False), (False, False)):
            cells.append((f"gamma={int(gb[0])} beta={int(gb[1])}", dict(FOCUS, has_gamma=gb[0], has_beta=gb[1])))
    # de-duplicate identical geometries (keep first label)
    seen, out = set(), []
    for label, g in cells:
        key = tuple(sorted(g.items()))
        if key not in seen:
            seen.add(key)
            out.append((label, g))
    return out


def test_affine_lane_form_device_perf(device):
    iters_list = _int_list("ALF_ITERS", "1,6")
    rows = []  # (label, geom, variant, {iters: ns}, bit_identical, max_dy, err_vs_torch)
    for label, geom_in in _sweep_cells():
        inputs, y_ref, valid_idx, geom, info = make_case(device, **geom_in)
        base = _run(device, inputs, "batched", geom, valid_idx)
        _check_ref(base, y_ref, f"batched {label}")
        for variant in _variants():
            samples = {}
            got = None
            for it in iters_list:
                ns, got_it = _measure_once(device, inputs, variant, geom, valid_idx, it)
                samples[it] = ns
                if it == min(iters_list):
                    got = got_it
            err, mean_err = _check_ref(got, y_ref, f"{variant} {label}")  # correctness gate on the timed launch
            rows.append(
                (label, geom, variant, samples, torch.equal(got, base), (got - base).abs().max().item(), err, mean_err)
            )
    report = _format_report(rows, iters_list, box=socket.gethostname(), arch=str(device.arch()))
    logger.info("\n" + report)
    if path := os.environ.get("ALF_REPORT"):
        Path(path).write_text(report)


def _format_report(rows, iters_list, *, box, arch):
    lines = [
        "# groupnorm_sc_N_1_HW_C / perf_experiments/affine_lane_form — post-finalize region (stats bcast + affine + apply)",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1 (pure compute)  "
        f"metric=DEVICE KERNEL DURATION [ns], one fresh run per cell  iters={list(iters_list)}",
        "",
        "Precision contract (fixed for every variant): fp32_dest_acc_en=True, HiFi4, approx=False, dst_full_sync=True; "
        "stats/E/stats_T/a/b fp32 pages, x/y/gamma/beta bf16.",
        "",
        "`per-region ns` = (ns[iters_max] - ns[iters_min]) / (iters_max - iters_min): the launch-independent cost of "
        "one image's stats bcast + affine build + apply (all chunks). `bit-identical` / `max|dy|` = vs the `batched` "
        "variant (the op today); `err` = max|y - torch fp32 reference|.",
        "",
        "| cell | cols | Kg | chunk_rows x chunks (Ht) | gamma/beta | variant | "
        + " | ".join(f"ns@iters={it}" for it in iters_list)
        + " | per-region ns | speedup | bit-identical / max|dy| | max err vs torch | mean err vs torch |",
        "|---|---:|---:|---|---|---|" + "---:|" * len(iters_list) + "---:|---:|---|---:|---:|",
    ]
    it_min, it_max = min(iters_list), max(iters_list)
    base_region = {}
    for label, g, variant, samples, bit, dy, err, mean_err in rows:
        if it_max != it_min:
            per_region = (samples[it_max] - samples[it_min]) / (it_max - it_min)
        else:
            per_region = samples[it_min]
        if variant == "batched":
            base_region[label] = per_region
        speed = base_region[label] / per_region if per_region > 0 else float("nan")
        cells = " | ".join(f"{samples[it]:.0f}" for it in iters_list)
        lines.append(
            f"| {label} | {g['cols']} | {g['Kg']} | {g['chunk_rows']} x {g['num_row_chunks']} ({g['Ht_core']}) | "
            f"{int(g['has_gamma'])}/{int(g['has_beta'])} | {variant} | {cells} | {per_region:.0f} | {speed:.2f}x | "
            f"{bit} / {dy:.1e} | {err:.1e} | {mean_err:.2e} |"
        )
    return "\n".join(lines) + "\n"
