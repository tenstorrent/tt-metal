# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness for the `fused_sumsq` idea.

Two honest variants of rms_norm's Regime B REDUCE PHASE, everything else held
byte-identical (same reader, same writer, same scale phase, same blocking, same
user precision config):

    baseline : per-chunk `square` -> cb_squared (a full-block intermediate CB)
               -> accumulating `reduce` over the whole chunk.  What the op does
               today.
    fused    : per-chunk `sum_of_squares` (x*x + per-row DEST accumulate, NO
               intermediate CB) -> ONE raw partial tile per row, folded across
               chunks by the SAME accumulating reduce over a 1-tile reduce dim.

Metric is `DEVICE KERNEL DURATION [ns]` out of the Tracy per-op CSV that
`scripts/run_safe_pytest.sh --profile` emits.  Precision is measured in the same
run (PCC + the row-scale bias `test_rms_norm_precision_matrix` uses), because a
faster sum of squares with a scale bias is a regression, not a win.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ttnn

import importlib.util as _ilu
from pathlib import Path as _Path

_loader = _ilu.module_from_spec(_ilu.spec_from_file_location('_fused_sumsq_loader', _Path(__file__).resolve().parent / '_load.py'))
_loader.__spec__.loader.exec_module(_loader)
rms_norm = _loader.load('fs_rms_norm').rms_norm

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

import os

N_WARMUP = 2
N_ITERS = int(os.environ.get("FS_ITERS", "5"))

MANIFEST_PATH = Path("generated/fused_sumsq_manifest.json")


def cfg_loose():
    """The focus case's EXACT precision corner (feature_spec.LOOSE_CASES perf cases).

    FROZEN: every variant runs under this identical descriptor.  No variant is
    allowed to move math_fidelity / fp32_dest_acc_en / math_approx_mode.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def cfg_default():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# name -> (shape, dtype, layout, gamma_layout)
SHAPES = {
    # THE focus case.
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime B on the full grid.
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime B + the masked partial reduce (W not tile-aligned).
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # The other perf-gated decode widths.
    "decode_2304": ((1, 1, 32, 2304), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_5120": ((1, 1, 32, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Smallest supported.
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime A control - must be untouched by this idea.
    "regimeA": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # The ROW_MAJOR input path (reduce phase preceded by a tilize).
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    # W-non-aligned on a MULTI-row-block plan (BLOCK_HT can exceed 1), which is
    # the only regime where the partial split needs the strided accumulate.
    "w_nonalign_tall": ((1, 1, 8192, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # bfloat8_b activations (interm/acc CB formats differ).  At (1,1,32,7168) a
    # bfloat8_b tile is 1088 B, so the resident plan FITS and this lands in
    # Regime A - it is a control, not a fused-path case.
    "bf8b": ((1, 1, 32, 7168), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # bfloat8_b wide enough to actually reach Regime B.
    "bf8b_B": ((1, 1, 8192, 7168), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # ROW_MAJOR input wide enough to reach Regime B (the fused reduce preceded by
    # a per-chunk tilize).  (1,1,8192,1024) RM fits Regime A, so it is the control.
    "row_major_B": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    # W-non-aligned on a NARROW width, so the L1 solver can grow BLOCK_HT past 1.
    # Only reachable in Regime B with fs_force_b - and it is the ONLY cell that
    # exercises the strided partial split ("every column but the last" is not
    # contiguous once a chunk spans more than one tile-row).
    "w_nonalign_bht": ((1, 1, 8192, 95), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "aligned_bht": ((1, 1, 8192, 96), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Where does the fused form stop paying?  Wt_core = 1 / 2 / 4, forced into
    # Regime B (they all fit the resident plan otherwise).  At Wt_core == 1 there
    # is nothing to fuse: one tile cannot have its square+reduce merged into a
    # cross-tile accumulate, so the accumulator bookkeeping is pure overhead.
    "wt1_B": ((1, 1, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "wt2_B": ((1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "wt4_B": ((1, 1, 32, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "wt1_partial_B": ((1, 1, 32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
}

# The candidate menu.  `baseline` MUST stay the op's current settings.
VARIANTS = {
    "baseline": dict(fused_sumsq=0),
    # Attribution arm: cb_squared stays allocated so the L1 solver lands on the
    # IDENTICAL blocking plan -> the delta is the compute payload alone.
    "fused": dict(fused_sumsq=1, fs_keep_squared=1),
    "fused_fold": dict(fused_sumsq=1, fs_keep_squared=1, fs_reload=1),
    # Shallower 16-bit-DEST accumulation (2 / 4 accumulators per W-chunk).
    "fused_g2": dict(fused_sumsq=1, fs_keep_squared=1, fs_group=-2),
    "fused_g4": dict(fused_sumsq=1, fs_keep_squared=1, fs_group=-4),
    # The real thing: cb_squared GONE, the freed L1 handed to the solver.
    "fused_l1": dict(fused_sumsq=1, fs_keep_squared=0),
    "fused_l1_fold": dict(fused_sumsq=1, fs_keep_squared=0, fs_reload=1),
    # --- COMPUTE-ONLY attribution (/perf-measure ablation) --------------------
    # stub_dm keeps every CB op + barrier in the reader/writer and issues no NoC
    # transfer, so the wall clock is the compute payload with zero starvation.
    # This is where the reduce phase's tile-op halving is visible: on the focus
    # shape the un-stubbed reduce phase waits on DRAM, which HIDES it.
    "baseline_sd": dict(fused_sumsq=0, stub_dm=1),
    "fused_sd": dict(fused_sumsq=1, fs_keep_squared=1, stub_dm=1),
    "fused_fold_sd": dict(fused_sumsq=1, fs_keep_squared=1, fs_reload=1, stub_dm=1),
    "fused_g2_sd": dict(fused_sumsq=1, fs_keep_squared=1, fs_group=-2, stub_dm=1),
    # --- Regime B forced (cells the L1 solver would route to Regime A) --------
    "baseline_fb": dict(fused_sumsq=0, fs_force_b=1),
    "fused_fb": dict(fused_sumsq=1, fs_keep_squared=1, fs_force_b=1),
    "fused_fold_fb": dict(fused_sumsq=1, fs_keep_squared=1, fs_reload=1, fs_force_b=1),
}


def resolve_levers(name, shape):
    """`fs_group=-N` means 'N accumulators per chunk' — resolved against the plan."""
    lev = dict(VARIANTS[name])
    g = lev.get("fs_group", 0)
    if g and g < 0:
        # The plan's own W-chunk is not known here; pass the DIVISOR request as a
        # cap and let the descriptor snap it to a divisor of WT_REDUCE_BLOCK.
        lev["fs_group"] = max(1, _wt_chunk(shape) // (-g))
    return lev


_WT_CACHE = {}


def _wt_chunk(shape):
    """WT_REDUCE_BLOCK the baseline plan picks for this shape (cached)."""
    return _WT_CACHE.get(tuple(shape), 1)


def prime_wt(device, name):
    """Record the baseline plan's WT_REDUCE_BLOCK so `fs_group=-N` can resolve."""
    fsd = _loader.load('fs_descriptor')
    shape, dtype, layout, glayout = SHAPES[name]
    x, g = make(device, shape, dtype, layout, glayout)
    plan = fsd.blocking_plan(x, g, x, device, cfg_loose(), None)
    _WT_CACHE[tuple(shape)] = plan.WT_REDUCE_BLOCK
    return plan


def make(device, shape, dtype, layout, gamma_layout):
    import torch

    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), dtype=dtype, layout=layout, device=device)
    g = ttnn.from_torch(
        torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else gamma_layout,
        device=device,
    )
    return x, g


def _dispatch(device, run_fn, iters=N_ITERS):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        run_fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def run_arm(device, manifest, name, variant, iters=N_ITERS, config="loose"):
    shape, dtype, layout, glayout = SHAPES[name]
    if tuple(shape) not in _WT_CACHE:
        prime_wt(device, name)
    x, g = make(device, shape, dtype, layout, glayout)
    cfg = cfg_loose() if config == "loose" else cfg_default()
    levers = resolve_levers(variant, shape)
    n = _dispatch(device, lambda: rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=levers), iters)
    manifest.append(
        {
            "label": f"{name}/{config}/{variant}",
            "shape": name,
            "variant": variant,
            "config": config,
            "levers": levers,
            "calls": n,
            "profiled": iters,
        }
    )


def write_manifest(manifest, path=MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def report_from_csv(csv_path, manifest_path=MANIFEST_PATH):
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = {}, 0
    for arm in manifest:
        i += arm["calls"] - arm["profiled"]
        window = rows[i : i + arm["profiled"]]
        i += arm["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in window if r.get(_DURATION_KEY))
        out[arm["label"]] = vals[len(vals) // 2] if vals else None
    return out


def print_report(csv_path, manifest_path=MANIFEST_PATH):
    rep = report_from_csv(csv_path, manifest_path)
    by_shape = {}
    for label, ns in rep.items():
        shape, config, variant = label.split("/")
        by_shape.setdefault((shape, config), {})[variant] = ns
    lines = []
    for (shape, config), variants in by_shape.items():
        base = variants.get("baseline")
        for v, ns in variants.items():
            sp = f"{base / ns:.3f}x" if (base and ns) else "-"
            lines.append(f"{shape:18s} {config:8s} {v:16s} {ns and int(ns):>10} ns   {sp}")
    print("\n".join(lines))
    return rep
