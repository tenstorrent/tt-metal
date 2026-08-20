# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""regime_b_resident: shapes, arm table and on-device runners.

BASELINE = `rbr_plan.BASELINE_LEVERS` — the lab plan with the ladder and the
masked-resident predicate switched OFF, which `rbr_plan.assert_matches_op_plan()`
gates as field-identical (CB layout included) to the op's own `blocking_plan()`.
Every number is a WITHIN-RUN delta against that arm: the lab compiles its kernels
with `-DRMSN_NO_ZONES`, so an absolute ns here is a few percent off the op's
zone-instrumented bench number.

MEASUREMENT: `DEVICE KERNEL DURATION [ns]` out of the Tracy per-op CSV that
`scripts/run_safe_pytest.sh --profile` writes.  Device kernel time has no warm-up
transient, so the warm-up dispatch exists ONLY to take JIT compilation out of the
window; the reported number is the median of N_ITERS.
"""

from __future__ import annotations

import json
from pathlib import Path

import ttnn

MANIFEST_PATH = Path("generated/regime_b_resident_manifest.json")
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 1
N_ITERS = 4

# name -> (shape, dtype, layout, gamma_layout)
SHAPES = {
    # --- MANDATORY no-regression arm: already handled by the shipped w_split ---
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- the guard-set masked representative: THE target of this idea ---------
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- more G=1 masked shapes ----------------------------------------------
    # 4127 -> Wt 130, W_partial 31 (a DIFFERENT partial count and a different Wt
    # factorization from 4095's Wt 128 / partial 31... 4095 % 32 = 31 too, so this
    # one moves Wt off a power of two, which is what the chunk search sees).
    "w_nonalign_4127": ((1, 1, 32, 4127), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # 6143 -> Wt 192: too wide for the FULL resident set, so this is where the
    # Regime C rung (x resident, scale chunked) has to carry it, masked.  Rt = 2.
    "w_nonalign_6143": ((1, 1, 64, 6143), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # the prefill twin of w_nonalign: Rt = 256 row-blocks over the grid.
    "w_nonalign_prefill": ((1, 1, 8192, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # MORE than one row-block per core - the regime where the second DRAM read
    # stops being hidden by the chunk pipeline.  512 / 256 row-blocks over a 130
    # core grid = 4 / 2 blocks per core.
    "w_nonalign_prefill4": ((1, 1, 16384, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "w_nonalign_prefill_6143": ((1, 1, 8192, 6143), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # the same >1-row-block regime at the OTHER two dtypes: the gate has to be a
    # work-distribution property, not a dtype list, so it must win here too.
    "prefill_4095_bf8b": ((1, 1, 8192, 4095), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "prefill_4095_fp32": ((1, 1, 8192, 4095), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # the SAME cell at the other DEST width: bf16 data, fp32_dest_acc_en=True.
    # That flips the masked reduce onto the OTHER datapath (ReduceTile +
    # `last_tile_at(1)` instead of AccumulateViaAdd + `partial_mask`), so it is a
    # distinct kernel path, not just a knob.
    "prefill_4095_destacc": ((1, 1, 8192, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # a masked width that is SMALL enough that the partial is most of the last
    # tile (W % 32 = 1) - the mask's extreme end.
    "w_nonalign_narrow": ((1, 1, 32, 1057), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- ALIGNED shapes the split cannot take (P2/P1), i.e. G = 1 by property --
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "wide_32768": ((1, 1, 32, 32768), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- the op's own bench shapes (controls / other datapaths) ---------------
    "grid_starved": ((1, 1, 32, 8192), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "rm_nonalign": ((1, 1, 1024, 4095), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- precision corners ----------------------------------------------------
    "prefill_1024_bf8b": ((1, 1, 8192, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "prefill_1024_fp32": ((1, 1, 8192, 1024), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "w_nonalign_bf8b": ((1, 1, 32, 4095), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "w_nonalign_fp32": ((1, 1, 32, 4095), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # --- PAD-POISON shapes (feature_spec's own list) --------------------------
    # THE gate on the masked resident fold.  At W = 4095 the implicit tile padding
    # is 31 of 4096 columns, so an unmasked reduce is wrong by only sqrt(4096/4095)
    # - 1 = 0.012%, which is BELOW bf16's own quantization step and therefore
    # invisible to PCC.  These widths make one tile of padding 11-38% of the row
    # AND fill it with 1000.0, so a leak is catastrophic (15-27% off) instead of
    # invisible.  Without them every "correct" number below would be vacuous.
    "poison_40": ((1, 1, 32, 40), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_72": ((1, 1, 32, 72), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_136": ((1, 1, 32, 136), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_200": ((1, 1, 32, 200), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_hw": ((1, 1, 40, 40), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # the guard-set width, poisoned: proves the mask lands on a WIDE masked shape
    # too, where the un-poisoned error would be invisible.
    "poison_4095": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_6143": ((1, 1, 64, 6143), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    # POISONED **AND** MULTI-ROW-BLOCK.  Under the blocks-per-core gate the small
    # poison shapes above all fall back to Regime B, so they no longer touch the
    # new path at all - these do.  They are the actual pad-poison gate on the
    # masked RESIDENT fold as it would ship.
    "poison_prefill_40": ((1, 1, 8192, 40), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_prefill_72": ((1, 1, 8192, 72), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_prefill_200": ((1, 1, 8192, 200), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
    "poison_prefill_4095": ((1, 1, 8192, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
}

# feature_spec's `_PAD_POISON_VALUE`.  Applied to the implicit tile padding of the
# input AND of a TILE gamma, exactly as the golden helper does.
PAD_POISON_VALUE = 1000.0
POISON_SHAPES = {n for n in SHAPES if n.startswith("poison_")}


def loose_cfg():
    """The focus case's EXACT precision corner.  FROZEN — never a perf lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def dest_acc_cfg():
    """Same fidelity, fp32_dest_acc_en=True — a DIFFERENT user contract, not a lever.

    Both arms of any comparison run under whichever of these the case names; the
    knob is never turned for speed.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def default_cfg():
    """The op's own default corner (HiFi4 / fp32_dest_acc_en=True)."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


CONFIGS = {"loose": loose_cfg, "dest_acc": dest_acc_cfg, "default": default_cfg}

# float32 activations cannot run at fp32_dest_acc_en=False in this op's contract
# surface, so that one case names the default corner; everything else is `loose`.
CASE_CONFIG = {
    "prefill_1024_fp32": "default",
    "w_nonalign_fp32": "default",
    "prefill_4095_fp32": "default",
    "prefill_4095_destacc": "dest_acc",
}


def config_for(name):
    return CASE_CONFIG.get(name, "loose")


# --- arm table ---------------------------------------------------------------
# Spelled literally rather than imported from rbr_plan: `ttnn.operations` walks
# and imports every .py under the operations tree, and a bare `import rbr_plan`
# only resolves once a caller has put this directory on sys.path.
BASELINE_LEVERS = dict(allow_c=0, allow_masked_resident=0, force_regime_lab="")

BASE = ("BASE", dict(BASELINE_LEVERS))

# The candidate: the ladder ON.  Everything else identical.
CAND = ("LADDER", dict())

DOMAIN_ARMS = [BASE, CAND]

# The pad-poison POSITIVE CONTROL.  Same plan as LADDER, mask removed.
POISON_CONTROL = ("NOMASK_CTRL", dict(resident_no_mask=1))


def focus_arms():
    """Decomposition + knob arms, for the shapes where the ladder actually lands."""
    arms = [BASE, CAND]
    # which RUNG did the ladder take?  Pin each explicitly so the win can be
    # attributed to "full resident" vs "x-only resident".
    arms.append(("A_masked", dict(force_regime_lab="A")))
    arms.append(("C_only", dict(allow_c=1, force_regime_lab="C")))
    arms.append(("C_chunked_gamma", dict(force_regime_lab="C", c_resident_gamma=0)))
    # chunk curve for Regime C at its default depth preference
    for c in (64, 32, 16, 8, 4, 2):
        arms.append((f"C_ws{c}", dict(force_regime_lab="C", c_ws=c)))
    # DEEPER: let the reader prefetch the NEXT row-block's x while the scale pass
    # is still reading this one out of L1.  Only means anything at > 1 row-block
    # per core - the single-row-block decode shapes have nothing to prefetch.
    for c in (32, 16, 8):
        arms.append((f"C_ws{c}_in2", dict(force_regime_lab="C", c_ws=c, c_in_depth=2, c_out_depth=2, c_gamma_depth=2)))
    # chunked gamma at a small chunk: hands the full-width gamma CB back to the
    # depth ladder, at the cost of re-reading gamma once per row-block.
    for c in (32, 16):
        arms.append((f"C_ws{c}_cg", dict(force_regime_lab="C", c_ws=c, c_resident_gamma=0)))
    return arms


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
    if name in POISON_SHAPES:
        # The torch reference is built from the LOGICAL tensor and never sees the
        # padding, so anything that folds pad into the reduce diverges hard.
        if layout == ttnn.TILE_LAYOUT:
            x = ttnn.fill_implicit_tile_padding(x, PAD_POISON_VALUE)
        if glayout == ttnn.TILE_LAYOUT:
            g = ttnn.fill_implicit_tile_padding(g, PAD_POISON_VALUE)
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
    """mean(computed_rms / reference_rms) - 1 — the systematic scale error PCC is blind to."""
    import torch

    ref_inv = torch.rsqrt(xt.pow(2).mean(dim=-1, keepdim=True) + eps)
    denom = xt * gt.reshape(*([1] * (xt.dim() - 1)), -1)
    mask = denom.abs() > 1e-2
    got_inv = torch.where(mask, got / torch.where(mask, denom, torch.ones_like(denom)), torch.zeros_like(denom))
    per_row = (got_inv.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).unsqueeze(-1)
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
                "bias": a.get("bias"),
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
        for r in rs:
            spd = f"{base / r['ns']:.3f}x" if (base and r["ns"]) else "-"
            print(f"  {r['label']:<34} {str(r['ns']):>12} ns  {spd:>8}  pcc={r['pcc']} bias={r['bias']}  {r['plan']}")
