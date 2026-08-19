# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off driver: rms_norm's scale pass, baseline vs fused arms.

Correctness is the ONLY assertion here; perf is measured and reported, never
asserted.

    # correctness of every arm, on every domain point
    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/fused_scale/test_fused_scale.py -k correct -s

    # the measurement (device kernel duration, one dispatch window per arm)
    scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/rms_norm/perf_experiments/fused_scale/test_fused_scale.py -k perf -s
    python3 -c "from ttnn.operations.rms_norm._bench_rms_norm import report_from_csv as r; \
        [print(f'{k:<48} {v[0]}') for k, v in r('<csv>', 'generated/fused_scale_manifest.json').items()]"

Env:
    FS_POINTS   comma list of domain-sweep point names (default: all)
    FS_ARMS     comma list of arm names (default: all)
    FS_ITERS    kernel_iters per dispatch (default 20)
    FS_PROF     profiled dispatches per arm (default 3).  The device profiler
                only dumps ~72 dispatches per run, so keep
                (points x arms x (1 + FS_PROF)) under that or the tail of the
                sweep is silently missing from the CSV (measured).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as opd
from ttnn.operations.rms_norm.perf_experiments.fused_scale import scale_bench as sb

MANIFEST_PATH = Path("generated/fused_scale_manifest.json")


def focus_config():
    """The focus case's exact precision corner (feature_spec LOOSE_CASES perf case):
    bf16 / HiFi2 / fp32_dest_acc_en=False.  FROZEN — an input to every arm."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


# name -> (shape, dtype, layout, gamma_layout, has_gamma)
# Every point's SCALE-PASS geometry is taken from the real op's own
# `blocking_plan`, so the bench runs the block shape the op actually emits.
POINTS = {
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "focus_no_gamma": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, False),
    "focus_bf8b": ((1, 1, 32, 7168), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, True),
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, True),
}

FUSED_ARMS = (
    "fused_rmsfull",
    "fused_inchain",
    "fused_gammafull",
    "fused_gammafull_amortized",
    "raw_llk",
    "fused_sfpu",
)


def plan_for(device, name):
    """The op's OWN plan for this shape — geometry, not a re-derivation."""
    shape, dtype, layout, gamma_layout, has_gamma = POINTS[name]
    x = SimpleNamespace(shape=list(shape), layout=layout, dtype=dtype)
    g = SimpleNamespace(shape=[1, 1, 1, shape[-1]], layout=gamma_layout, dtype=dtype) if has_gamma else None
    plan = opd.blocking_plan(x, g, None, device, focus_config())
    return plan, dtype, has_gamma


def geometry(plan):
    return dict(wt=plan.WT_SCALE_BLOCK, block_ht=plan.BLOCK_HT, dest_block=plan.DEST_BLOCK)


# SUB_CHUNK values for the baseline_subchunk arm: how few cb_normed pages the
# two-mul form can be squeezed into, and what the extra per-call overhead costs.
SUB_CHUNKS = (8, 16, 32, 56)


def arms_for(geo, cfg):
    """(label, arm, dest_block, sub_chunk).  The reuse arms are capped at the
    largest block that is CORRECT (scale_bench.REUSE_ARMS), and the baseline is
    measured at BOTH its own DEST_BLOCK (what the op runs today) and the capped
    one, so the fusion is never credited with a block-size difference."""
    d = geo["dest_block"]
    r = sb.dest_block_for("fused_rmsfull", d, cfg)
    arms = [("baseline", "baseline", d, 0), ("baseline_reversed", "baseline_reversed", d, 0)]
    if r != d:
        arms.append((f"baseline@blk{r}", "baseline", r, 0))
    arms += [(f"{a}@blk{r}", a, r, 0) for a in FUSED_ARMS]
    arms += [
        (f"baseline_subchunk{s}", "baseline_subchunk", d, s) for s in SUB_CHUNKS if s < geo["wt"] and s >= d
    ]
    sel = os.environ.get("FS_ARMS")
    if sel:
        keep = [k for k in sel.split(",") if k]
        arms = [a for a in arms if a[1] in keep]
    return arms


def _points():
    sel = os.environ.get("FS_POINTS")
    return [s for s in sel.split(",") if s] if sel else list(POINTS)


@pytest.mark.parametrize("point", _points())
def test_correct(device, point):
    """Every arm must reproduce x * 1/rms * gamma.  A faster wrong answer is out."""
    import torch

    plan, dtype, has_gamma = plan_for(device, point)
    geo = geometry(plan)
    cfg = focus_config()
    x, gamma, rms, ref = sb.make_inputs(device, wt=geo["wt"], block_ht=geo["block_ht"], dtype=dtype, has_gamma=has_gamma)

    print(
        f"\n[{point}] regime={plan.regime} Wt_core={plan.Wt_core} BLOCK_HT={geo['block_ht']} "
        f"WT_SCALE_BLOCK={geo['wt']} DEST_BLOCK={geo['dest_block']} "
        f"num_row_blocks={plan.num_row_blocks} scale_chunks={plan.Wt_core // geo['wt']}"
    )
    failures = []
    for label, arm, blk, sub in arms_for(geo, cfg):
        out = sb.run_arm(
            x, gamma, rms, arm=arm, wt=geo["wt"], block_ht=geo["block_ht"], dest_block=blk, kernel_iters=1,
            compute_kernel_config=cfg, sub_chunk=sub,
        )
        got = ttnn.to_torch(out).float()
        pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        l1_b = sb.scratch_bytes(
            arm, dtype=dtype, has_gamma=has_gamma, wt=geo["wt"], block_ht=geo["block_ht"], sub_chunk=sub
        )
        print(f"  {label:<34} blk={blk} pcc={pcc:.6f}  max_rel={rel:.4g}  interm_L1={l1_b:>8} B")
        if not (pcc > 0.9995):
            failures.append((label, pcc))
        ttnn.deallocate(out)
    assert not failures, f"arms produced the wrong answer: {failures}"


def test_perf(device):
    """One dispatch window per (point, arm); DEVICE KERNEL DURATION comes off Tracy."""
    iters = int(os.environ.get("FS_ITERS", "20"))
    cfg = focus_config()
    manifest = []
    for point in _points():
        plan, dtype, has_gamma = plan_for(device, point)
        geo = geometry(plan)
        x, gamma, rms, _ = sb.make_inputs(
            device, wt=geo["wt"], block_ht=geo["block_ht"], dtype=dtype, has_gamma=has_gamma
        )
        for label, arm, blk, sub in arms_for(geo, cfg):
            run = lambda a=arm, b=blk, u=sub: ttnn.deallocate(
                sb.run_arm(
                    x, gamma, rms, arm=a, wt=geo["wt"], block_ht=geo["block_ht"], dest_block=b,
                    kernel_iters=iters, compute_kernel_config=cfg, sub_chunk=u,
                )
            )
            run()  # warm-up dispatch (kernel build / cache), not profiled
            ttnn.synchronize_device(device)
            n_prof = int(os.environ.get("FS_PROF", "3"))
            for _ in range(n_prof):
                run()
            ttnn.synchronize_device(device)
            manifest.append(
                {
                    "label": f"{point}/{label}",
                    "shape": point,
                    "levers": {
                        "arm": arm,
                        "blk": blk,
                        "sub_chunk": sub,
                        "iters": iters,
                        "wt": geo["wt"],
                        "block_ht": geo["block_ht"],
                        "regime": plan.regime,
                        "tiles_per_iter": geo["wt"] * geo["block_ht"],
                        "interm_L1_bytes": sb.scratch_bytes(
                            arm, dtype=dtype, has_gamma=has_gamma, wt=geo["wt"], block_ht=geo["block_ht"],
                            sub_chunk=sub,
                        ),
                    },
                    "calls": 1 + n_prof,
                    "profiled": n_prof,
                }
            )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nFUSED_SCALE: manifest -> {MANIFEST_PATH} ({len(manifest)} arms, kernel_iters={iters})")
    for a in manifest:
        print(f"  {a['label']:<48} {a['levers']}")
    assert manifest
