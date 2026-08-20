# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""fused_sumsq bake-off driver.  Correctness is the only pass/fail; perf is measured.

    # correctness + precision of every variant across the domain sweep
    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/rms_norm/perf_experiments/fused_sumsq/test_fused_sumsq.py \\
        -k correctness -s

    # the 16-bit-DEST sum-of-squares bias sweep (Wt = 32/64/128/224, Regime B forced)
    scripts/run_safe_pytest.sh --run-all ... -k bias -s

    # the profiled ns menu
    scripts/run_safe_pytest.sh --profile ... -k perf -s
    python3 -c "from ttnn.operations.rms_norm.perf_experiments.fused_sumsq import fs_bench as b; \\
                b.print_report('<csv>')"
"""

import os

import pytest

import ttnn

# torch is imported lazily inside each function: ttnn/ forbids a module-level
# torch import (pre-commit `check-torch-imports-in-ttnn`), and this module lives
# under the operations tree.

import importlib.util as _ilu
from pathlib import Path as _Path

_loader = _ilu.module_from_spec(
    _ilu.spec_from_file_location("_fused_sumsq_loader", _Path(__file__).resolve().parent / "_load.py")
)
_loader.__spec__.loader.exec_module(_loader)
bench = _loader.load("fs_bench")
fsd = _loader.load("fs_descriptor")
rms_norm = _loader.load("fs_rms_norm").rms_norm

EPS = 1e-6

# Soft gate from the focus case's `extras` (feature_spec LOOSE_CASES).
PCC_SOFT = 0.9995
# A row scale is either right or structurally wrong; rounding never reaches 2%.
MAX_ROW_SCALE_BIAS = 0.02

SWEEP = [
    "focus",
    "prefill_7168",
    "w_nonalign",
    "w_nonalign_tall",
    "decode_2304",
    "decode_5120",
    "smallest",
    "regimeA",
    "row_major",
    "row_major_B",
    "bf8b",
    "bf8b_B",
]

MENU = ["baseline", "fused", "fused_fold", "fused_g2", "fused_g4", "fused_l1", "fused_l1_fold"]
# stub_dm arms are perf-only (no correct output by construction).
PERF_MENU = MENU + ["baseline_sd", "fused_sd", "fused_fold_sd", "fused_g2_sd"]


def _row_scale_bias(xg, out, s_ref):
    """Mean relative error of the per-row 1/rms factor (least-squares fit out ~ k*xg)."""
    gf = xg.reshape(-1, xg.shape[-1])
    of = out.reshape(-1, out.shape[-1])
    k = (of * gf).sum(-1) / (gf * gf).sum(-1).clamp_min(1e-30)
    return ((k / s_ref.reshape(-1)) - 1.0).mean().item()


def _measure(device, shape_name, variant, config="loose", extra_levers=None):
    import torch

    shape, dtype, layout, glayout = bench.SHAPES[shape_name]
    W = shape[-1]
    if tuple(shape) not in bench._WT_CACHE:
        bench.prime_wt(device, shape_name)
    x, g = bench.make(device, shape, dtype, layout, glayout)
    cfg = bench.cfg_loose() if config == "loose" else bench.cfg_default()
    levers = bench.resolve_levers(variant, shape)
    if extra_levers:
        levers.update(extra_levers)

    plan = fsd.blocking_plan(x, g, x, device, cfg, levers)

    x32 = ttnn.to_torch(x).float()
    g32 = ttnn.to_torch(g).float().reshape(1, 1, 1, -1)[..., :W]
    s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    expected = x32 * s_ref * g32

    actual = ttnn.to_torch(rms_norm(x, gamma=g, epsilon=EPS, compute_kernel_config=cfg, _levers=levers)).float()

    pcc = torch.corrcoef(torch.stack([expected.flatten().double(), actual.flatten().double()]))[0, 1].item()
    bias = _row_scale_bias(x32 * g32, actual, s_ref)
    return plan, pcc, bias


@pytest.mark.parametrize("variant", MENU)
@pytest.mark.parametrize("shape_name", SWEEP)
def test_correctness(device, shape_name, variant):
    plan, pcc, bias = _measure(device, shape_name, variant)
    print(
        f"\nFS_CORRECT shape={shape_name} variant={variant} regime={plan.regime} "
        f"Wt_core={plan.Wt_core} WT_REDUCE={plan.WT_REDUCE_BLOCK} BLOCK_HT={plan.BLOCK_HT} "
        f"FS_GROUP={plan.FS_GROUP} W_partial={plan.W_partial} via_add={plan.reduce_via_add} "
        f"ws_bytes={plan.working_set_bytes()} pcc={pcc:.6f} bias={bias:+.5f}"
    )
    assert abs(bias) < MAX_ROW_SCALE_BIAS, f"row-scale bias {bias:+.5f} >= {MAX_ROW_SCALE_BIAS}"
    assert pcc >= PCC_SOFT, f"pcc {pcc:.6f} < {PCC_SOFT}"


# --- Regime B forced ----------------------------------------------------------
# Cells the L1 solver would otherwise route to Regime A (or to BLOCK_HT == 1), so
# the fused path is exercised on them at all: bfloat8_b activations, and the
# W-non-aligned x BLOCK_HT > 1 corner that needs the STRIDED partial split.
FORCED_B = ["w_nonalign_bht", "aligned_bht", "bf8b", "decode_2304", "regimeA", "row_major"]


@pytest.mark.parametrize("variant", MENU)
@pytest.mark.parametrize("shape_name", FORCED_B)
def test_forced_b_correctness(device, shape_name, variant):
    plan, pcc, bias = _measure(device, shape_name, variant, extra_levers=dict(fs_force_b=1))
    print(
        f"\nFS_FORCEB shape={shape_name} variant={variant} regime={plan.regime} "
        f"Wt_core={plan.Wt_core} WT_REDUCE={plan.WT_REDUCE_BLOCK} BLOCK_HT={plan.BLOCK_HT} "
        f"FS_GROUP={plan.FS_GROUP} W_partial={plan.W_partial} via_add={plan.reduce_via_add} "
        f"pcc={pcc:.6f} bias={bias:+.5f}"
    )
    assert abs(bias) < MAX_ROW_SCALE_BIAS, f"row-scale bias {bias:+.5f} >= {MAX_ROW_SCALE_BIAS}"
    assert pcc >= PCC_SOFT, f"pcc {pcc:.6f} < {PCC_SOFT}"


# --- the DEFAULT precision corner (fp32_dest_acc_en=True, HiFi4) --------------
# There the op picks the ReduceTile datapath (reduce_via_add gates on 16-bit
# DEST), so this is the OTHER datapath the fused form has to be correct on: each
# per-chunk raw partial tile is collapsed within-tile and the REDUCED partials are
# accumulated - which is exactly what ReduceTile's cross-call Accumulate does.
DEFAULT_CORNER = ["focus", "w_nonalign", "smallest", "decode_5120", "row_major_B", "regimeA"]


@pytest.mark.parametrize("variant", ["baseline", "fused", "fused_fold", "fused_l1"])
@pytest.mark.parametrize("shape_name", DEFAULT_CORNER)
def test_default_corner_correctness(device, shape_name, variant):
    plan, pcc, bias = _measure(device, shape_name, variant, config="default", extra_levers=dict(fs_force_b=1))
    print(
        f"\nFS_DEFCFG shape={shape_name} variant={variant} regime={plan.regime} "
        f"Wt_core={plan.Wt_core} WT_REDUCE={plan.WT_REDUCE_BLOCK} BLOCK_HT={plan.BLOCK_HT} "
        f"W_partial={plan.W_partial} via_add={plan.reduce_via_add} acc={plan.acc_dtype} "
        f"pcc={pcc:.6f} bias={bias:+.5f}"
    )
    assert abs(bias) < MAX_ROW_SCALE_BIAS, f"row-scale bias {bias:+.5f} >= {MAX_ROW_SCALE_BIAS}"
    assert pcc >= PCC_SOFT, f"pcc {pcc:.6f} < {PCC_SOFT}"


# --- the 16-bit-DEST sum-of-squares bias sweep --------------------------------
# Exactly the axis the op's own rationale block documents (+0.84% at Wt=32 ->
# +10.4% at Wt=224 on the ReduceTile datapath).  Regime B is FORCED so the narrow
# widths exercise the same datapath as the focus shape.
BIAS_SHAPES = {32: (1, 1, 32, 1024), 64: (1, 1, 32, 2048), 128: (1, 1, 32, 4096), 224: (1, 1, 32, 7168)}


@pytest.mark.parametrize("variant", ["baseline", "fused", "fused_fold", "fused_g2", "fused_g4"])
@pytest.mark.parametrize("wt", [32, 64, 128, 224])
def test_bias(device, wt, variant):
    import torch

    shape = BIAS_SHAPES[wt]
    W = shape[-1]
    torch.manual_seed(0)
    cfg = bench.cfg_loose()  # fp32_dest_acc_en=False — the 16-bit DEST corner
    x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.randn(1, 1, 1, W), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    levers = dict(bench.VARIANTS[variant])
    levers["fs_force_b"] = 1
    plan0 = fsd.blocking_plan(x, g, x, device, cfg, dict(levers, fs_group=0))
    gsel = levers.get("fs_group", 0)
    if gsel and gsel < 0:
        levers["fs_group"] = max(1, plan0.WT_REDUCE_BLOCK // (-gsel))
    plan = fsd.blocking_plan(x, g, x, device, cfg, levers)

    x32 = ttnn.to_torch(x).float()
    g32 = ttnn.to_torch(g).float().reshape(1, 1, 1, -1)[..., :W]
    s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    actual = ttnn.to_torch(rms_norm(x, gamma=g, epsilon=EPS, compute_kernel_config=cfg, _levers=levers)).float()
    expected = x32 * s_ref * g32
    pcc = torch.corrcoef(torch.stack([expected.flatten().double(), actual.flatten().double()]))[0, 1].item()
    bias = _row_scale_bias(x32 * g32, actual, s_ref)
    print(
        f"\nFS_BIAS Wt={wt} variant={variant} regime={plan.regime} WT_REDUCE={plan.WT_REDUCE_BLOCK} "
        f"FS_GROUP={plan.FS_GROUP} pcc={pcc:.6f} row_scale_bias={bias:+.5f}"
    )


# --- the profiled ns menu -----------------------------------------------------
@pytest.mark.timeout(3600)
def test_perf(device):
    shapes = os.environ.get("FS_PERF_SHAPES", ",".join(SWEEP)).split(",")
    variants = os.environ.get("FS_PERF_VARIANTS", ",".join(PERF_MENU)).split(",")
    manifest = []
    for name in [s for s in shapes if s]:
        for v in [x for x in variants if x]:
            bench.run_arm(device, manifest, name, v)
    path = bench.write_manifest(manifest)
    print(f"\nFS_MANIFEST {path}")
    assert manifest


# --- L1 accounting (host-only, no device math) --------------------------------
def test_l1_accounting(device):
    for name in ["focus", "prefill_7168", "w_nonalign", "decode_2304", "decode_5120"]:
        shape, dtype, layout, glayout = bench.SHAPES[name]
        x, g = bench.make(device, shape, dtype, layout, glayout)
        cfg = bench.cfg_loose()
        rows = {}
        for v in ["baseline", "fused", "fused_l1"]:
            lev = dict(bench.VARIANTS[v])
            lev.pop("fs_group", None)
            p = fsd.blocking_plan(x, g, x, device, cfg, lev)
            rows[v] = p
        b, f, fl = rows["baseline"], rows["fused"], rows["fused_l1"]
        sq = dict((i, n * pb) for i, n, pb, _ in b.cb_layout).get(fsd.CB_SQUARED, 0)
        acc = dict((i, n * pb) for i, n, pb, _ in f.cb_layout).get(fsd.CB_SUMSQ_ACC, 0)
        print(
            f"\nFS_L1 shape={name} regime={b.regime} Wt_core={b.Wt_core} "
            f"base_ws={b.working_set_bytes()} base_cb_squared={sq} fused_cb_sumsq_acc={acc} "
            f"net_freed={sq - acc} | fused_l1: WT_REDUCE={fl.WT_REDUCE_BLOCK} (base {b.WT_REDUCE_BLOCK}) "
            f"IN_DEPTH={fl.IN_BUF_DEPTH} (base {b.IN_BUF_DEPTH}) BLOCK_HT={fl.BLOCK_HT} (base {b.BLOCK_HT}) "
            f"ws={fl.working_set_bytes()} budget={fl.l1_cb_budget}"
        )
