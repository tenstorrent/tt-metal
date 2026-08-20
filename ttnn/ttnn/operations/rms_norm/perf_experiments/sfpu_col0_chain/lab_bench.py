# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""sfpu_col0_chain WHOLE-OP bake-off: shape/arm table, dispatch + manifest, report.

Metric: `DEVICE KERNEL DURATION [ns]` from the Tracy per-op CSV that
`scripts/run_safe_pytest.sh --profile` emits.  Device kernel time has no warm-up
transient, so each arm issues a fixed number of dispatches after two untimed
warm-ups and the report takes the MEDIAN of that window.

Every arm runs under the SAME user precision config; no arm touches
math_fidelity, fp32_dest_acc_en, math_approx_mode, dst_full_sync_en or a dtype.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import ttnn

from .lab_descriptor import blocking_plan
from .lab_op import default_compute_kernel_config, lab_rms_norm, loose_compute_kernel_config

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 2
N_ITERS = int(os.environ.get("SC_ITERS", "10"))

MANIFEST_PATH = Path(os.environ.get("SC_MANIFEST", "generated/sfpu_col0_chain_manifest.json"))

TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT

# name -> (shape, dtype, layout, gamma_dtype, gamma_layout, config)
SHAPES = {
    # ---- the perf-gated focus case (feature_spec LOOSE_CASES), EXACT config ----
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- prefill: grid-filling, many row-blocks per core (the chain runs often) --
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- smallest supported shape: per-core-overhead regime ---------------------
    "smallest": ((32, 17), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- non-tile-aligned W: masked reduce / partial-scaler datapath ------------
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- the other activation dtypes; fp32 runs at fp32 DEST (different SFPU
    #      datapath: DST_ACCUM_MODE flips the body's bf16 convert off) ------------
    "prefill_1024_bf8b": ((1, 1, 8192, 1024), ttnn.bfloat8_b, TILE, ttnn.bfloat8_b, TILE, "loose"),
    "prefill_1024_fp32": ((1, 1, 8192, 1024), ttnn.float32, TILE, ttnn.float32, TILE, "default"),
    # ---- Regime B (streaming, W-chunked) via the forced-regime lever ------------
    "regime_b_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- ROW_MAJOR input + RM gamma: the tilize/untilize datapath ---------------
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, RM, ttnn.bfloat16, RM, "loose"),
    # ---- H non-aligned (phantom-row clamp) --------------------------------------
    "h_nonalign": ((1, 1, 100, 736), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- no gamma: the chain's consumer writes straight to the output CB --------
    "no_gamma": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, None, None, "loose"),
}

# shapes that need an extra lever regardless of arm (regime forcing)
SHAPE_LEVERS = {"regime_b_1024": {"force_regime": 1}}

CONFIGS = {"default": default_compute_kernel_config, "loose": loose_compute_kernel_config}

# arm id -> levers.  `baseline` is the op's CURRENT approach, verbatim
# (chain_scope 0 == the untouched `ckl::eltwise_chain` call).
ARMS = {
    "baseline": {"chain_scope": 0},
    "vmode_c": {"chain_scope": 1},
    "cskip": {"chain_scope": 2},
    "fused_rc": {"chain_scope": 3},
    "fused_c": {"chain_scope": 4},
    "fused_cskip": {"chain_scope": 5},
}

# DON'T-CARE PROBES (not candidates): NaN-stamp every DEST lane before the chain.
# The stamp covers every lane EXCEPT column 0 and runs AFTER the chain.
#   poison_cskip  - the candidate scope.  MUST still pass the torch gate.
#   poison_base   - the baseline scope, same stamp.  MUST also pass (control:
#                   the poison, not the scope, is what is being isolated).
POISON_ARMS = {
    "poison_cskip": {"chain_scope": 6},
    "poison_base": {"chain_scope": 7},
}
# POSITIVE CONTROL, kept OUT of the gated set: the same probe with column 0
# poisoned too.  It MUST fail the torch gate; if it passes, the probe lands
# nowhere and the two poison arms above are passing vacuously.
POISON_POSITIVE_CONTROL = {"poison_all": {"chain_scope": 8}}

ALL_ARMS = dict(ARMS, **POISON_ARMS, **POISON_POSITIVE_CONTROL)


def make_tensors(device, name):
    import torch

    shape, dtype, layout, g_dtype, g_layout, _ = SHAPES[name]
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=dtype, layout=layout, device=device)
    if g_dtype is None:
        return x, None, xt, None
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    g = ttnn.from_torch(gt, dtype=g_dtype, layout=g_layout, device=device)
    return x, g, xt, gt


def torch_reference(xt, gt, eps=1e-6):
    import torch

    x = xt.to(torch.float32)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if gt is not None:
        y = y * gt.to(torch.float32).reshape(-1)
    return y


def pcc(a, b):
    import torch

    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    if not torch.isfinite(a).all():
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float("nan") if d == 0 else (a @ b / d).item()


def row_scale_bias(got, ref):
    """Systematic % error in the OUTPUT SCALE, which PCC is blind to.

    Per row, the least-squares scale `s` with got ~= s*ref; the reported number
    is mean(s) - 1 in percent.  The op's changelog tracks this because a reduce
    datapath can bias the rms denominator uniformly without moving PCC.
    """
    import torch

    g = got.to(torch.float32).reshape(-1, got.shape[-1])
    r = ref.to(torch.float32).reshape(-1, ref.shape[-1])
    num = (g * r).sum(-1)
    den = (r * r).sum(-1)
    s = num / den.clamp_min(1e-30)
    return float((s.mean() - 1.0) * 100.0)


def plan_of(device, name):
    shape, dtype, layout, g_dtype, g_layout, cfg = SHAPES[name]
    x, g, _, _ = make_tensors(device, name)
    return blocking_plan(x, g, x, device, CONFIGS[cfg](), levers=SHAPE_LEVERS.get(name))


def _dispatch(device, run_fn, iters=N_ITERS):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        run_fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def levers_for(name, arm):
    lv = dict(SHAPE_LEVERS.get(name, {}))
    lv.update(ALL_ARMS[arm])
    return lv


def run_arm(device, manifest, name, arm, iters=N_ITERS):
    cfg = CONFIGS[SHAPES[name][5]]()
    x, g, _, _ = make_tensors(device, name)
    lv = levers_for(name, arm)
    n = _dispatch(device, lambda: lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=lv), iters)
    manifest.append({"label": f"{name}/{arm}", "shape": name, "arm": arm, "levers": lv, "calls": n, "profiled": iters})


def write_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return MANIFEST_PATH


def print_report(csv_path, manifest_path=MANIFEST_PATH):
    """Fold the Tracy per-op CSV back onto the manifest labels BY POSITION."""
    import statistics

    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get(_DURATION_KEY, "").strip()]
    durations = [float(r[_DURATION_KEY]) for r in rows]

    i, out = 0, {}
    for entry in manifest:
        i += N_WARMUP
        window = durations[i : i + entry["profiled"]]
        i += entry["profiled"]
        if window:
            out[entry["label"]] = statistics.median(window)
    if i != len(durations):
        print(f"WARNING: manifest expects {i} dispatches, CSV has {len(durations)}")

    shapes = list(dict.fromkeys(e["shape"] for e in manifest))
    arms = list(dict.fromkeys(e["arm"] for e in manifest))
    print(f"\n{'shape':<20} " + " ".join(f"{a:>13}" for a in arms))
    for s in shapes:
        base = out.get(f"{s}/baseline")
        cells = []
        for a in arms:
            v = out.get(f"{s}/{a}")
            cells.append("-".rjust(13) if v is None else f"{v:>8.0f}ns")
        print(f"{s:<20} " + " ".join(cells))
        if base:
            sp = []
            for a in arms:
                v = out.get(f"{s}/{a}")
                sp.append("-".rjust(13) if v is None else f"{base / v:>12.3f}x")
            print(f"{'  (speedup)':<20} " + " ".join(sp))
    return out
