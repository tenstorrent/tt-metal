# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + on-device measurement for perf_experiments/affine_batch (idea E1).

Correctness is the only pass/fail: every variant must (1) match a torch reference of the affine build within fp32
tolerance and (2) be compared against the baseline variant's output (bit-identical or max-abs-diff reported).
Perf is measured (DEVICE KERNEL DURATION [ns], in-process device profiler), never asserted.

Run:  scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/perf_experiments/affine_batch/test_affine_batch.py
Env:  AB_COLS="1,2,4,5,8"  AB_KG="1,2"  AB_ITERS="1,11"  AB_REPORT=<path>
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

from ttnn.operations.groupnorm_sc_N_1_HW_C.perf_experiments.affine_batch.affine_batch_bench import (
    TILE,
    VARIANTS,
    membership_tile_index,
    run_affine,
    sharded_row_config,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


# =============================================================================
# Synthetic inputs (one column group of `cols` channel tiles, Kg group tiles)
# =============================================================================
def _group_size(cols, Kg):
    """Channels per group: the first Cg >= cols*32/G that does NOT divide 32, so groups straddle tile boundaries
    (a channel tile then holds parts of several groups). Channels >= G*Cg (if any) are padded lanes (E = 0)."""
    G = Kg * TILE
    Cg = max(1, math.ceil(cols * TILE / G))
    while 32 % Cg == 0:
        Cg += 1
    return Cg


def make_case(device, cols, Kg, seed=11):
    torch.manual_seed(seed)
    G = Kg * TILE
    Cg = _group_size(cols, Kg)
    C_valid = min(G * Cg, cols * TILE)
    mean_g = torch.rand(G) * 2.0 - 1.0  # distinct per-lane values
    rstd_g = torch.rand(G) * 1.5 + 0.5
    # stats_g_full: tiles [mean_kg0..mean_kg(Kg-1), rstd_kg0..]; lane g&31 of tile kg = g>>5, all 32 rows equal
    stats = torch.zeros(TILE, 2 * Kg * TILE)
    for g in range(G):
        kg, gl = g >> 5, g & 31
        stats[:, kg * TILE + gl] = mean_g[g]
        stats[:, (Kg + kg) * TILE + gl] = rstd_g[g]
    # membership E_T[g_local, c] = 1 if channel T*32+c is in group g (kg = g>>5), per-variant tile order
    E_logical = torch.zeros(cols, Kg, TILE, TILE)  # [tl, kg, row=g_local, col=c]
    for ch in range(C_valid):
        tl, c = ch // TILE, ch % TILE
        g = ch // Cg
        E_logical[tl, g >> 5, g & 31, c] = 1.0
    gamma = torch.randn(cols * TILE).to(torch.bfloat16)
    beta = torch.randn(cols * TILE).to(torch.bfloat16)
    gamma_row = torch.zeros(TILE, cols * TILE, dtype=torch.bfloat16)
    beta_row = torch.zeros(TILE, cols * TILE, dtype=torch.bfloat16)
    gamma_row[0] = gamma
    beta_row[0] = beta
    # reference: mean_T[c] = sum_g E[g,c] mean_g ; a = rstd_T * gamma ; b = beta - mean_T * a   (fp32)
    mean_T = torch.zeros(cols * TILE)
    rstd_T = torch.zeros(cols * TILE)
    for ch in range(C_valid):
        g = ch // Cg
        mean_T[ch] = mean_g[g]
        rstd_T[ch] = rstd_g[g]
    a_ref = (rstd_T * gamma.to(torch.float32)).unsqueeze(0).expand(TILE, -1).contiguous()
    b_ref = (
        (beta.to(torch.float32) - mean_T * rstd_T * gamma.to(torch.float32)).unsqueeze(0).expand(TILE, -1).contiguous()
    )

    def to_dev(t, dtype, n_tiles):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_row_config(n_tiles)
        )

    def membership_for(variant):
        E = torch.zeros(TILE, cols * Kg * TILE)
        for tl in range(cols):
            for kg in range(Kg):
                idx = membership_tile_index(variant, tl, kg, cols, Kg)
                E[:, idx * TILE : (idx + 1) * TILE] = E_logical[tl, kg]
        return to_dev(E, ttnn.float32, cols * Kg)

    inputs = {
        "stats": to_dev(stats, ttnn.float32, 2 * Kg),
        "membership": {v: membership_for(v) for v in VARIANTS},
        "gamma": to_dev(gamma_row, ttnn.bfloat16, cols),
        "beta": to_dev(beta_row, ttnn.bfloat16, cols),
    }
    refs = (mean_T, rstd_T, gamma.to(torch.float32), beta.to(torch.float32))
    return inputs, (a_ref, b_ref), dict(G=G, Cg=Cg, C_valid=C_valid, refs=refs)


def _run(device, inputs, variant, cols, Kg, iters=1, zones=False):
    a, b = run_affine(
        device,
        inputs["stats"],
        inputs["membership"][variant],
        inputs["gamma"],
        inputs["beta"],
        variant=variant,
        cols=cols,
        Kg=Kg,
        iters=iters,
        zones=zones,
    )
    return ttnn.to_torch(a).to(torch.float32), ttnn.to_torch(b).to(torch.float32)


def _check_ref(got, ref, label):
    # Sanity vs torch only: the FPU evaluates the affine at tf32-class operand precision (the op documents this
    # in its compute kernel), so a relative error of ~2^-11 is inherent to the op's approach. The real gate is the
    # per-variant comparison against the baseline variant (bit-identical / max-abs-diff), reported per case.
    logger.info(f"{label}: max|got-ref| = {(got - ref).abs().max().item():.3e}")
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2, msg=f"{label} vs torch reference")


def _round_tf32(x):
    # fp32 -> 10 explicit mantissa bits (round-to-nearest-even on the dropped 13 bits)
    bits = x.contiguous().view(torch.int32)
    lsb = (bits >> 13) & 1
    bits = (bits + 0x0FFF + lsb) & ~0x1FFF
    return bits.view(torch.float32)


def _precision_hypotheses(base_a, base_b, refs):
    """Which operand rounding reproduces the device's affine build? (documents the op's inherent precision)."""
    mean_T, rstd_T, gamma, beta = refs
    a_dev = base_a[0]
    for name, h in (
        ("exact", lambda t: t),
        ("tf32", _round_tf32),
        ("bf16", lambda t: t.to(torch.bfloat16).to(torch.float32)),
    ):
        a_h = h(rstd_T) * gamma
        b_h = beta - h(h(mean_T) * h(a_dev))
        logger.info(
            f"  hypothesis {name:5s}: max|a_dev - a_h| = {(a_dev - a_h).abs().max().item():.3e}   "
            f"max|b_dev - b_h| = {(base_b[0] - b_h).abs().max().item():.3e}"
        )


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


def _measure_once(device, inputs, variant, cols, Kg, iters):
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # consume anything pending
    _run(device, inputs, variant, cols, Kg, iters=iters)
    ns = _read_kernel_ns(device)
    assert ns is not None, f"no profiler data for {variant} cols={cols} Kg={Kg} iters={iters}"
    return ns


def _int_list(name, default):
    return tuple(int(v) for v in os.environ.get(name, default).split(","))


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("Kg", [1, 2])
@pytest.mark.parametrize("cols", [1, 2, 4, 5, 8])
def test_affine_batch_correctness(device, cols, Kg):
    inputs, (a_ref, b_ref), info = make_case(device, cols, Kg)
    base_a, base_b = _run(device, inputs, "baseline", cols, Kg)
    _check_ref(base_a, a_ref, f"baseline a cols={cols} Kg={Kg}")
    _check_ref(base_b, b_ref, f"baseline b cols={cols} Kg={Kg}")
    _precision_hypotheses(base_a, base_b, info.pop("refs"))
    for variant in VARIANTS[1:]:
        a, b = _run(device, inputs, variant, cols, Kg)
        _check_ref(a, a_ref, f"{variant} a cols={cols} Kg={Kg}")
        _check_ref(b, b_ref, f"{variant} b cols={cols} Kg={Kg}")
        da = (a - base_a).abs().max().item()
        db = (b - base_b).abs().max().item()
        logger.info(
            f"cols={cols} Kg={Kg} {info} {variant}: a bit-identical={torch.equal(a, base_a)} max|da|={da:.3e} ; "
            f"b bit-identical={torch.equal(b, base_b)} max|db|={db:.3e}"
        )
    # the multi-iteration kernel (used for the per-group slope) must produce the same output as iters=1
    for variant in VARIANTS:
        a, b = _run(device, inputs, variant, cols, Kg, iters=3)
        _check_ref(a, a_ref, f"{variant} a iters=3 cols={cols} Kg={Kg}")
        _check_ref(b, b_ref, f"{variant} b iters=3 cols={cols} Kg={Kg}")


def test_affine_batch_device_perf(device):
    cols_list = _int_list("AB_COLS", "1,2,4,5,8")
    kg_list = _int_list("AB_KG", "1,2")
    iters_list = _int_list("AB_ITERS", "1,11")
    rows = []  # (cols, Kg, variant, {iters: ns}, bit_a, bit_b, max_da, max_db)
    for Kg in kg_list:
        for cols in cols_list:
            inputs, (a_ref, b_ref), info = make_case(device, cols, Kg)
            info.pop("refs")
            base_a, base_b = _run(device, inputs, "baseline", cols, Kg)
            _check_ref(base_a, a_ref, "baseline a")
            _check_ref(base_b, b_ref, "baseline b")
            for variant in VARIANTS:
                a, b = _run(device, inputs, variant, cols, Kg)  # correctness gate before timing
                _check_ref(a, a_ref, f"{variant} a cols={cols} Kg={Kg}")
                _check_ref(b, b_ref, f"{variant} b cols={cols} Kg={Kg}")
                samples = {it: _measure_once(device, inputs, variant, cols, Kg, it) for it in iters_list}
                rows.append(
                    (
                        cols,
                        Kg,
                        variant,
                        samples,
                        torch.equal(a, base_a),
                        torch.equal(b, base_b),
                        (a - base_a).abs().max().item(),
                        (b - base_b).abs().max().item(),
                    )
                )
    report = _format_report(rows, iters_list, box=socket.gethostname(), arch=str(device.arch()))
    logger.info("\n" + report)
    if path := os.environ.get("AB_REPORT"):
        Path(path).write_text(report)


def _format_report(rows, iters_list, *, box, arch):
    lines = [
        "# groupnorm_sc_N_1_HW_C / perf_experiments/affine_batch — pass-2 affine build per column group",
        "",
        f"box={box}  arch={arch}  cores=1  placement=single-core sharded-L1 (pure compute)  "
        f"metric=DEVICE KERNEL DURATION [ns], one fresh run per cell  iters={list(iters_list)}",
        "",
        "Precision contract (fixed for every variant): fp32_dest_acc_en=True, HiFi4, approx=False, dst_full_sync=True; "
        "stats/E fp32 pages, gamma/beta bf16 row tiles, a/b fp32 full tiles.",
        "",
        "`per-group ns` = (ns[iters_max] - ns[iters_min]) / (iters_max - iters_min): the launch-independent cost of one "
        "column group's affine build. `per-tile ns` = per-group / cols. `bit-identical` = vs the baseline variant's output.",
        "",
        "| cols | Kg | variant | "
        + " | ".join(f"ns@iters={it}" for it in iters_list)
        + " | per-group ns | per-tile ns | speedup (per-group) | a bit-identical / max|da| | b bit-identical / max|db| |",
        "|---:|---:|---|" + "---:|" * len(iters_list) + "---:|---:|---:|---|---|",
    ]
    it_min, it_max = min(iters_list), max(iters_list)
    base_group = {}
    for cols, Kg, variant, samples, bit_a, bit_b, da, db in rows:
        if it_max != it_min:
            per_group = (samples[it_max] - samples[it_min]) / (it_max - it_min)
        else:
            per_group = samples[it_min]
        if variant == "baseline":
            base_group[(cols, Kg)] = per_group
        speed = base_group[(cols, Kg)] / per_group if per_group > 0 else float("nan")
        cells = " | ".join(f"{samples[it]:.0f}" for it in iters_list)
        lines.append(
            f"| {cols} | {Kg} | {variant} | {cells} | {per_group:.0f} | {per_group / cols:.0f} | {speed:.2f}x | "
            f"{bit_a} / {da:.1e} | {bit_b} / {db:.1e} |"
        )
    return "\n".join(lines) + "\n"
