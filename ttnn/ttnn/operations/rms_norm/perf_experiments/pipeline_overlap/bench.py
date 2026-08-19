# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pipeline_overlap arm table + on-device runners.

THE IDEA UNDER TEST
-------------------
On the focus shape the op's plan is "coarsest W-chunk (112 of 224 tiles), every
streaming CB at depth 1".  The cumulative ablation says the reader and the
compute stages are ADDITIVE (full 43,983 ~= stub_dm 28,221 + stub_compute 24,093
- floor 9,230), i.e. nothing overlaps: with a depth-1 input CB the reader cannot
reserve chunk c+1 until compute has popped chunk c.  This sweeps the
(chunk size x per-CB depth) surface to find the assignment that actually
pipelines, and measures it against that plan.

L1 ARITHMETIC (focus shape, bf16, BLOCK_HT=1, W_partial=0, all pages 2048 B)
    pages(wr, ws) = d_in*max(wr,ws) + d_sq*wr + (d_g + d_n + d_out)*ws + 4
    budget        = 1,269,888 B = 620 pages
so at wr = ws = c the depth budget is c*(d_in+d_sq+d_g+d_n+d_out) <= 616:
    c=112 -> sum <= 5   (== the op's 1,1,1,1,1: the coarsest chunk cannot buffer)
    c=224 -> sum <= 2   (INEXPRESSIBLE - not even depth 1 fits; this is why the
                         op's solver settled on 112)
    c=56  -> sum <= 11
    c=28  -> sum <= 22
Every arm below is a point on that surface, plus the split-chunk arms that give
the reduce pass and the scale pass DIFFERENT granularities.
"""

from __future__ import annotations

import json
from pathlib import Path

import ttnn

MANIFEST_PATH = Path("generated/pipeline_overlap_manifest.json")
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 1
N_ITERS = 4

# name -> (shape, dtype, layout, gamma_layout)
SHAPES = {
    # THE focus case (feature_spec LOOSE_CASES perf case), Wt_core = 224.
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime B on the full grid, DRAM-bound.
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime B + masked partial reduce, Wt_core = 128 (NOT a power-of-two multiple
    # of the chunk candidates by accident - the CB-wrap gate).
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Smallest supported shape - per-core overhead regime.
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime A control: single-chunk by construction, must be untouched.
    "regime_a": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Regime B, ROW_MAJOR gamma (staging CB + compute-side tilize) - the other
    # gamma datapath, to check the gamma-depth finding is not TILE-gamma-only.
    "rm_gamma": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
}


def loose_cfg():
    """The focus case's EXACT precision corner.  Frozen: never a perf lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _arm(label, wr, ws=None, din=1, dout=1, dg=1, dsq=1, dn=1, **extra):
    ws = wr if ws is None else ws
    lev = dict(
        wt_reduce=wr,
        wt_scale=ws,
        in_depth=din,
        out_depth=dout,
        gamma_depth=dg,
        squared_depth=dsq,
        normed_depth=dn,
    )
    lev.update(extra)
    return (label, lev)


def _pages(wr, ws, din, dout, dg, dsq, dn):
    return din * max(wr, ws) + dsq * wr + (dg + dn + dout) * ws + 4


PAGE_BUDGET = 620  # 1,269,888 B / 2048 B, focus shape


def focus_arms():
    """Every (chunk x depth) point that fits L1 on the focus shape (Wt_core=224)."""
    arms = []
    divisors = [224, 112, 56, 32, 28, 16, 14, 8, 4, 2]

    # --- the honest baseline: exactly the op's current plan ------------------
    arms.append(_arm("BASE_c112_d1", 112))

    # --- S1: the chunk curve at depth 1 (the op's family, finer chunks) ------
    for c in divisors:
        if c == 112:
            continue
        if _pages(c, c, 1, 1, 1, 1, 1) <= PAGE_BUDGET:
            arms.append(_arm(f"S1_c{c}_d1", c))

    # --- S2: symmetric depth 2 on in / out / gamma --------------------------
    for c in divisors:
        if _pages(c, c, 2, 2, 2, 1, 1) <= PAGE_BUDGET:
            arms.append(_arm(f"S2_c{c}_in2out2g2", c, din=2, dout=2, dg=2))

    # --- S3: reader running further ahead ------------------------------------
    for c in divisors:
        if _pages(c, c, 4, 2, 2, 1, 1) <= PAGE_BUDGET:
            arms.append(_arm(f"S3_c{c}_in4out2g2", c, din=4, dout=2, dg=2))

    # --- S4: everything deep, including the compute->compute CBs -------------
    for c in divisors:
        if _pages(c, c, 4, 4, 4, 2, 2) <= PAGE_BUDGET:
            arms.append(_arm(f"S4_c{c}_all4", c, din=4, dout=4, dg=4, dsq=2, dn=2))

    # --- S5: WHICH CB wants the depth, at a fixed chunk (c=56, budget sum 11) -
    for lbl, kw in (
        ("in4", dict(din=4)),
        ("out4", dict(dout=4)),
        ("g4", dict(dg=4)),
        ("in3out3", dict(din=3, dout=3)),
        ("in4g4", dict(din=4, dg=4)),
        ("in3out3g3", dict(din=3, dout=3, dg=3)),
        ("sq2n2", dict(dsq=2, dn=2)),
    ):
        assert _pages(56, 56, kw.get("din", 1), kw.get("dout", 1), kw.get("dg", 1), kw.get("dsq", 1), kw.get("dn", 1)) <= PAGE_BUDGET, lbl
        arms.append(_arm(f"S5_c56_{lbl}", 56, **kw))

    # --- S6: SPLIT chunks - the reduce pass and the scale pass differ --------
    for lbl, kw in (
        ("wr112_ws56_g2out2", dict(wr=112, ws=56, dg=2, dout=2)),
        ("wr112_ws28_g2n2out4", dict(wr=112, ws=28, dg=2, dn=2, dout=4)),
        ("wr112_ws56_in2g2out2", dict(wr=112, ws=56, din=2, dg=2, dout=2)),
        ("wr56_ws112_sq2", dict(wr=56, ws=112, dsq=2)),
        # wr=224 is the WHOLE width in one reduce chunk (zero reduce-pass
        # overlap, zero chunk overhead) paired with a finely-pipelined scale pass.
        ("wr224_ws28_g2out3", dict(wr=224, ws=28, dg=2, dn=1, dout=3)),
    ):
        p = _pages(kw["wr"], kw["ws"], kw.get("din", 1), kw.get("dout", 1), kw.get("dg", 1), kw.get("dsq", 1), kw.get("dn", 1))
        if p <= PAGE_BUDGET:
            arms.append(_arm(f"S6_{lbl}", **kw))

    # --- R: refinement round around the round-1 winner (c=56, deep gamma) -----
    # Round 1 says the depth that matters is on cb_gamma_tiles (c=56: gamma-only
    # depth 4 = 1.19x, input-only depth 4 = 1.02x, output-only depth 4 = 0.94x),
    # so this round walks the gamma depth and the chunk with the L1 that frees up.
    for lbl, kw in (
        ("c56_in2g2out1", dict(wr=56, din=2, dg=2, dout=1)),
        ("c56_in2g4out1", dict(wr=56, din=2, dg=4, dout=1)),
        ("c56_in2g4out2", dict(wr=56, din=2, dg=4, dout=2)),
        ("c56_in2g6out1", dict(wr=56, din=2, dg=6, dout=1)),
        ("c56_in1g7out1", dict(wr=56, din=1, dg=7, dout=1)),
        ("c56_in2g3out3", dict(wr=56, din=2, dg=3, dout=3)),
        ("c56_in2g2out2sq2n2", dict(wr=56, din=2, dg=2, dout=2, dsq=2, dn=2)),
        ("c32_in2g8out2", dict(wr=32, din=2, dg=8, dout=2)),
        ("c32_in4g8out4", dict(wr=32, din=4, dg=8, dout=4)),
        ("c28_in4g8out4", dict(wr=28, din=4, dg=8, dout=4)),
        ("c28_in2g12out2", dict(wr=28, din=2, dg=12, dout=2)),
        ("c16_in4g16out4", dict(wr=16, din=4, dg=16, dout=4)),
        ("c112_g2_only", dict(wr=112, dg=2)),  # expected INEXPRESSIBLE (676 > 620)
    ):
        p = _pages(kw["wr"], kw["wr"], kw.get("din", 1), kw.get("dout", 1), kw.get("dg", 1), kw.get("dsq", 1), kw.get("dn", 1))
        if p <= PAGE_BUDGET:
            arms.append(_arm(f"R_{lbl}", **kw))
        else:
            print(f"  [arm table] R_{lbl} INEXPRESSIBLE: {p} pages > {PAGE_BUDGET}")
    return arms


def policy_arms():
    """The candidate stated as a RULE, so it runs on any shape (domain sweep).

    BASE_op_plan runs the op's own solver verbatim through the lab descriptor
    (levers = {}), which `assert_matches_op_plan` proves is the same plan - that
    is the honest baseline on every shape, including Regime A and (32,17) where
    there is nothing to chunk.
    """
    return [
        ("BASE_op_plan", {}),
        # THE CANDIDATE.  Round-2 winner, stated as a rule: coarsest chunk at
        # which cb_input_tiles AND cb_gamma_tiles both reach depth 2.  Deeper
        # than 2 on either, and any depth on cb_output_tiles / cb_squared /
        # cb_normed, is measured flat (see the R group).
        ("POL_win", dict(policy_depths=(2, 1, 2, 1, 1))),
        ("POL_d2", dict(policy_depths=(2, 2, 2, 1, 1))),
        ("POL_in4", dict(policy_depths=(4, 2, 2, 1, 1))),
        ("POL_all4", dict(policy_depths=(4, 4, 4, 2, 2))),
        ("POLSPLIT_d2", dict(policy_split=(1, 1, 2, 1, 2))),
        ("POLSPLIT_d4", dict(policy_split=(1, 1, 2, 2, 4))),
    ]


# The subset carried into the domain sweep + the zone breakdown.  Filled in from
# the focus surface once it is measured; kept as a name list so the domain run is
# a one-line change.
DOMAIN_ARMS = ["BASE_op_plan", "POL_win", "POL_d2"]

# Cumulative /perf-measure ablation, run at BOTH the baseline plan and the
# winner, so the new bound classification is measured rather than inferred.
def ablation_arms():
    out = []
    for tag, base in (("BASE", dict(wt_reduce=112, wt_scale=112)), ("WIN", dict(policy_depths=(2, 1, 2, 1, 1)))):
        for abl, kw in (
            ("full", {}),
            ("stub_dm", dict(stub_dm=1)),
            ("stub_compute", dict(stub_compute=1)),
            ("stub_both", dict(stub_dm=1, stub_compute=1)),
        ):
            lev = dict(base)
            lev.update(kw)
            out.append((f"ABL_{tag}_{abl}", lev))
    return out


# --- device runners -----------------------------------------------------------


def make_tensors(device, name, seed=0):
    import torch

    shape, dtype, layout, glayout = SHAPES[name]
    torch.manual_seed(seed)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=dtype, layout=layout, device=device)
    g = ttnn.from_torch(gt, dtype=dtype, layout=glayout, device=device)
    return x, g, xt, gt


def torch_ref(xt, gt, eps=1e-6):
    import torch

    inv = torch.rsqrt(xt.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xt * inv * gt.reshape(1, 1, 1, -1)


def pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def dispatch(device, fn, iters=N_ITERS):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def write_manifest(manifest, path=MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def report_from_csv(csv_path, manifest_path=MANIFEST_PATH):
    """Fold the Tracy per-op CSV back onto the manifest labels, by dispatch order."""
    import csv as _csv

    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in _csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = [], 0
    for a in manifest:
        i += a["calls"] - a["profiled"]
        win = rows[i : i + a["profiled"]]
        i += a["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in win if r.get(_DURATION_KEY))
        out.append((a["label"], a["shape"], vals[len(vals) // 2] if vals else None, a.get("plan", ""), a.get("pcc")))
    return out
