# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gamma_row0 isolated bake-off: shape/arm table, dispatch + manifest, report.

Measurement discipline (/perf-measure): the metric is
`DEVICE KERNEL DURATION [ns]` out of the Tracy per-op CSV that
`scripts/run_safe_pytest.sh --profile` emits.  Device kernel time has no warm-up
transient, so each arm issues a small fixed number of dispatches and the report
takes the MEDIAN of that window; a trial loop would only re-measure the same
number.  Two untimed warm-up dispatches per arm keep JIT compilation out of the
window.

Every arm runs under the SAME user precision config (the `loose` corner:
bf16 / HiFi2 / fp32_dest_acc_en=False, which is what every perf-gated
feature_spec case supplies).  No arm touches math_fidelity, fp32_dest_acc_en,
math_approx_mode, dst_full_sync_en or a dtype.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ttnn

from .lab_descriptor import blocking_plan
from .lab_op import default_compute_kernel_config, lab_rms_norm, loose_compute_kernel_config

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 2
N_ITERS = 6

MANIFEST_PATH = Path("generated/gamma_row0_manifest.json")

TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT

# name -> (shape, dtype, layout, gamma_dtype, gamma_layout, config)
# The domain sweep. `focus` is the mandatory primary target.
SHAPES = {
    # ---- the perf-gated focus case (feature_spec LOOSE_CASES) --------------
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- prefill: many cores, DRAM-bandwidth regime -----------------------
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- narrow decode: a fixed gamma setup cost would show here -----------
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- smallest supported shape: per-core-overhead regime ----------------
    "smallest": ((32, 17), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- non-tile-aligned W: the last gamma tile's valid columns change ----
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- controls: OTHER gamma / input datapaths this must not regress -----
    "grid_starved": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, RM, "loose"),  # RM gamma
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, RM, ttnn.bfloat16, RM, "loose"),  # RM input
    "no_gamma": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, None, None, "loose"),
    # ---- the other gamma dtypes (own axis, independent of the activations) --
    "focus_g_fp32": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.float32, TILE, "loose"),
    "focus_g_bf8b": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.bfloat8_b, TILE, "loose"),
    # ---- H non-aligned (phantom-row clamp) and rank 3, both with TILE gamma ---
    "h_nonalign": ((1, 1, 100, 736), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    "rank3": ((3, 64, 2048), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- the fp32-activation corner (default config; fp32+dest_off excluded) --
    "prefill_1024_fp32": ((1, 1, 8192, 1024), ttnn.float32, TILE, ttnn.float32, TILE, "default"),
    "prefill_1024_bf8b": ((1, 1, 8192, 1024), ttnn.bfloat8_b, TILE, ttnn.bfloat8_b, TILE, "loose"),
}

CONFIGS = {"default": default_compute_kernel_config, "loose": loose_compute_kernel_config}

# arm id -> levers.  `baseline` is the op's CURRENT approach, verbatim.
ARMS = {
    "baseline": {},
    "span": {"gamma_read": 1},
    "faces": {"gamma_read": 2},
    # correctness probes: the untouched tile rows 1..31 stamped with NaN
    "baseline_poison": {"gamma_read": 0, "gamma_prefill": 2},
    "span_poison": {"gamma_read": 1, "gamma_prefill": 2},
    "faces_poison": {"gamma_read": 2, "gamma_prefill": 2},
    # option 2 (Regime B, TILE gamma): one DRAM gamma read per core + L1 refill
    "cache": {"gamma_read": 2, "gamma_cache": 1},
    # one-packet issue path (same bytes, shorter RISC issue sequence)
    "span_1pkt": {"gamma_read": 5},
    "faces_1pkt": {"gamma_read": 6},
    "full_1pkt": {"gamma_read": 7},  # control: isolates the issue-path effect
}

# ABLATION (not a candidate, deliberately incorrect): CB lifecycle + barrier kept,
# no gamma transfer issued.  Measures the FLOOR a perfect gamma ingest could reach,
# which is what bounds the multicast option before building it.
ABLATE = {"gamma_read": 4}

# Ceiling probe for option 3 (mcast): span read on ONE core only.  Also deliberately
# incorrect - it is a bound, not a candidate.
INJECT_ONLY = {"gamma_read": 8}

# NEGATIVE CONTROL, kept out of ARMS so the correctness gate does not assert on it:
# run 0 only (columns 0-15 of every gamma tile).  It MUST fail the torch gate; if it
# passes, the partial-read path is not actually in effect and every "win" is a
# measurement of the baseline.
NEG_CONTROL = {"gamma_read": 3}

# Everything the BENCH may dispatch.  ARMS is the correctness-gated candidate set;
# these two extras are measurement/diagnostic only and are never candidates.
ALL_ARMS = dict(ARMS, ablate=ABLATE, neg=NEG_CONTROL, inject_only=INJECT_ONLY)


def make_tensors(device, name, dtype_override=None):
    import torch

    shape, dtype, layout, g_dtype, g_layout, _ = SHAPES[name]
    dtype = dtype_override or dtype
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

    y = xt * torch.rsqrt(xt.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gt is not None:
        y = y * gt.reshape(-1)
    return y


def pcc(a, b):
    import torch

    # float64: at 8192x7168 a float32 dot/mean-centering drifts enough to report
    # a "correlation" above 1.0, which would mask a real regression.
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item() / denom)


def plan_of(device, name):
    x, g, _, _ = make_tensors(device, name)
    cfg = CONFIGS[SHAPES[name][5]]()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config()
    )
    return blocking_plan(x, g, out, device, cfg, None)


def _dispatch(device, fn, iters=N_ITERS):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def run_arm(device, manifest, name, arm, iters=N_ITERS, extra_levers=None, dtype_override=None):
    x, g, _, _ = make_tensors(device, name, dtype_override)
    cfg = CONFIGS[SHAPES[name][5]]()
    levers = dict(ALL_ARMS[arm])
    if extra_levers:
        levers.update(extra_levers)
    n = _dispatch(device, lambda: lab_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=levers), iters)
    manifest.append(
        {
            "label": f"{name}/{arm}" + (f"+{extra_levers}" if extra_levers else ""),
            "shape": name,
            "arm": arm,
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
    """Fold the Tracy per-op CSV onto the manifest labels, by dispatch order."""
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
    r = report_from_csv(csv_path, manifest_path)
    by_shape = {}
    for label, ns in r.items():
        shape, arm = label.split("/", 1)
        by_shape.setdefault(shape, {})[arm] = ns
    print(f"{'shape':<20} {'arm':<24} {'ns':>12} {'vs baseline':>12}")
    for shape, arms in by_shape.items():
        base = arms.get("baseline")
        for arm, ns in arms.items():
            rel = f"{base / ns:.3f}x" if (base and ns) else ""
            print(f"{shape:<20} {arm:<24} {ns:>12.0f} {rel:>12}")
    return r
