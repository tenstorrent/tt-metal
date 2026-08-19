# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""resident_single_read: arm table, shapes and on-device runners.

BASELINE = `allow_c=0` — the lab plan with Regime C switched off, which
`rsr_plan.assert_matches_op_plan()` gates as field-identical to the op's own
`blocking_plan()`.  Everything is measured against that, not against a number
carried in from another run: the lab compiles the kernels with `-DRMSN_NO_ZONES`,
so an absolute ns here is a few percent off the op's zone-instrumented bench
number and only the WITHIN-RUN delta is meaningful.

MEASUREMENT: `DEVICE KERNEL DURATION [ns]` out of the Tracy per-op CSV that
`scripts/run_safe_pytest.sh --profile` writes; median of N_ITERS dispatches after
one warm-up (the warm-up only exists to take JIT compilation out of the window —
device kernel time has no warm-up transient).
"""

from __future__ import annotations

import json
from pathlib import Path

import ttnn

MANIFEST_PATH = Path("generated/resident_single_read_manifest.json")
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 1
N_ITERS = 4

# name -> (shape, dtype, layout, gamma_layout).  Every one of these is either a
# feature_spec LOOSE_CASES perf case, an op bench shape, or a deliberate
# boundary/control probe (noted).
SHAPES = {
    # --- THE focus case: Regime B today, Wt_core=224, one core (Rt=1) --------
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # Same shape, ROW_MAJOR gamma (staging CB + compute-side tilize): the OTHER
    # gamma datapath, which Regime C has to drive per chunk.
    "focus_rmg": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    # --- the big absolute number D20's 50% is about --------------------------
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- other Regime-B decode widths ---------------------------------------
    "decode_5120": ((1, 1, 32, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_4096": ((1, 1, 32, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- the wide LOOSE_CASES: 16384 still fits resident-x, 32768 does NOT ---
    #     (the graceful-degradation pair — this is the crossover)
    "wide_16384": ((1, 1, 32, 16384), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "wide_32768": ((1, 1, 32, 32768), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- masked / non-aligned W: forced to B for a CORRECTNESS reason --------
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- Regime A controls (must be untouched) ------------------------------
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # ROW_MAJOR input path (tilize/untilize in the loop).  `row_major` is the op
    # bench shape and is Regime A already (control); `rm_wide` is wide enough
    # that the op picks B, so it is where Regime C actually lands on the RM path.
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "rm_wide": ((1, 1, 512, 7168), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    # --- precision-matrix widths (Wt = 32 / 64 / 128 / 224) -----------------
    "prec_wt64": ((1, 1, 32, 2048), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- other dtypes on a Regime-B width -----------------------------------
    "bf8b_7168": ((1, 1, 32, 7168), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "fp32_7168": ((1, 1, 32, 7168), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
}


def loose_cfg():
    """The focus case's EXACT precision corner.  FROZEN — never a perf lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def default_cfg():
    """The op's own default corner (HiFi4 / fp32_dest_acc_en=True)."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


CONFIGS = {"loose": loose_cfg, "default": default_cfg}

# --- arm table ---------------------------------------------------------------
# label -> levers.  BASE is the op's plan; every C_* arm keeps the SAME precision
# config and differs only in the blocking plan.
BASE = ("BASE", dict(allow_c=0))


def focus_arms(divisors=(112, 56, 32, 28, 16, 8, 4, 2, 1)):
    """The (scale-chunk x CB-depth) surface of Regime C on Wt_core = 224."""
    arms = [BASE]
    # the solver's own pick
    arms.append(("C_auto", dict()))
    # chunk curve at depth 1
    for c in divisors:
        arms.append((f"C_ws{c}_d1", dict(c_ws=c, c_in_depth=1, c_out_depth=1, c_gamma_depth=1, c_normed_depth=1)))
    # a few depth points at the chunks that leave L1 room
    for c in (56, 32, 28, 16, 8):
        arms.append((f"C_ws{c}_g2out2", dict(c_ws=c, c_in_depth=1, c_out_depth=2, c_gamma_depth=2, c_normed_depth=1)))
    for c in (32, 16, 8):
        arms.append(
            (f"C_ws{c}_g2out4n2", dict(c_ws=c, c_in_depth=1, c_out_depth=4, c_gamma_depth=2, c_normed_depth=2))
        )
    # the decomposition arm: resident x, but B's streaming reduce datapath
    for c in (112, 56, 32):
        arms.append((f"C_ws{c}_nofuse", dict(c_ws=c, c_fused_reduce=0)))
    # deepen the RESIDENT x CB too: only buys anything when a core owns more than
    # one row-block (the reader can prefetch row-block b+1 during the scale pass
    # of b), so it is a prefill arm, not a focus arm.
    arms.append(("C_in2", dict(c_in_depth=2, c_out_depth=2, c_gamma_depth=2)))
    # CANDIDATE 1 of the task list: resident x AND resident gamma, only
    # cb_normed + the output CB chunked.  Costs Wt_core gamma pages instead of
    # ws, and reads gamma once per CORE instead of once per row-block.
    arms.append(("C_rg", dict(c_resident_gamma=1)))
    arms.append(("C_rg_in2", dict(c_resident_gamma=1, c_in_depth=2, c_out_depth=2)))
    # Prices the single-read fast path from the other side: Regime B forced onto a
    # shape that fits Regime A (lever_ledger D20's stated follow-up).
    arms.append(("FORCE_B", dict(allow_c=0, force_regime="B")))
    return arms


# The domain sweep runs only these (the ns table per shape).
DOMAIN_ARMS = ["BASE", "C_auto", "C_in2", "C_rg", "C_rg_in2"]


# --- device runners ----------------------------------------------------------
def make_tensors(device, name, seed=0):
    import torch

    shape, dtype, layout, glayout = SHAPES[name]
    torch.manual_seed(seed)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    if dtype == ttnn.bfloat8_b:
        glayout = ttnn.TILE_LAYOUT
    x = ttnn.from_torch(xt, dtype=dtype, layout=layout, device=device)
    g = ttnn.from_torch(gt, dtype=dtype, layout=glayout, device=device)
    return x, g, xt, gt


def torch_ref(xt, gt, eps=1e-6):
    import torch

    inv = torch.rsqrt(xt.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xt * inv * gt.reshape(*([1] * (xt.dim() - 1)), -1)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def row_scale_bias(got, xt, gt, eps=1e-6):
    """mean(computed_rms / reference_rms) - 1, per the op's precision matrix.

    `got` is the kernel output; dividing it by (x*gamma) recovers 1/rms per row,
    so the ratio of reference-rms to that is the datapath's rms estimate.
    """
    import torch

    ref_inv = torch.rsqrt(xt.pow(2).mean(dim=-1, keepdim=True) + eps)
    denom = xt * gt.reshape(*([1] * (xt.dim() - 1)), -1)
    mask = denom.abs() > 1e-2
    got_inv = torch.where(mask, got / torch.where(mask, denom, torch.ones_like(denom)), torch.zeros_like(denom))
    per_row = (got_inv.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).unsqueeze(-1)
    # computed_rms / reference_rms = ref_inv / computed_inv
    ratio = ref_inv / per_row.clamp(min=1e-30)
    return float(ratio.mean().item() - 1.0)


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
        out.append(
            {
                "label": a["label"],
                "shape": a["shape"],
                "ns": vals[len(vals) // 2] if vals else None,
                "plan": a.get("plan", ""),
                "pcc": a.get("pcc"),
            }
        )
    return out


def print_report(rows):
    by_shape = {}
    for r in rows:
        by_shape.setdefault(r["shape"], []).append(r)
    for shape, rs in by_shape.items():
        base = next((r["ns"] for r in rs if r["label"].endswith("/BASE")), None)
        print(f"\n=== {shape} ===   baseline = {base}")
        for r in sorted(rs, key=lambda r: (r["ns"] is None, r["ns"] or 0)):
            spd = f"{base / r['ns']:.3f}x" if (base and r["ns"]) else "-"
            print(f"  {r['label']:<40} {str(r['ns']):>12} ns  {spd:>8}  pcc={r['pcc']}  {r['plan']}")
