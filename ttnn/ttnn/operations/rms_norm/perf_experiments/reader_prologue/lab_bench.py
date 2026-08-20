# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""reader_prologue isolated bake-off: shape/arm table, dispatch + manifest, report.

THE IDEA.  `rms_norm_reader.cpp` issues, strictly in this order, (1) the reduce
scaler tile, (2) the whole resident Regime A gamma slice, (3) the input chunk.
The compute thread cannot start `sum_of_squares` until the INPUT lands, so on the
focus shape ~950 ns of scaler + gamma latency sits at the HEAD of the critical
path - even though gamma is consumed by the LAST compute phase and the scaler by
the root's combine.  Every arm here is a pure REORDERING of that prologue.

MEASUREMENT (/perf-measure): the metric is `DEVICE KERNEL DURATION [ns]` out of
the Tracy per-op CSV that `scripts/run_safe_pytest.sh --profile` emits.  Device
kernel time has no warm-up transient, so each arm issues a small fixed number of
dispatches and the report takes the MEDIAN of that window.

PRECISION CONTRACT: every arm runs under the SAME user config (the `loose`
corner: bf16 / HiFi2 / fp32_dest_acc_en=False, exactly what the perf-gated
feature_spec cases supply).  No arm touches math_fidelity, fp32_dest_acc_en,
math_approx_mode, dst_full_sync_en or a dtype - the only knob that moves is
`prologue`, which changes the ORDER of transfers, never their size or format.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ttnn

from .lab_descriptor import blocking_plan
from .lab_op import default_compute_kernel_config, lab_rms_norm, loose_compute_kernel_config

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

import os

N_WARMUP = int(os.environ.get("RP_WARMUP", 2))
N_ITERS = int(os.environ.get("RP_ITERS", 6))

MANIFEST_PATH = Path("generated/reader_prologue_manifest.json")

TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT

# name -> (shape, dtype, layout, gamma_dtype, gamma_layout, config)
# Mirrors `_bench_rms_norm.py`'s BENCH_SHAPES / BENCH_GAMMA_LAYOUT verbatim: a
# shape that is absent from BENCH_GAMMA_LAYOUT gets ROW_MAJOR gamma there, which
# is a DIFFERENT datapath (staging CB + compute-side tilize), so it is spelled
# out here rather than defaulted.
SHAPES = {
    # ---- the perf-gated focus case (feature_spec LOOSE_CASES), TILE gamma ----
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- prefill: full grid, DRAM-bandwidth regime --------------------------
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- narrow decode: the per-core prologue is the largest share here ------
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    "decode_5120": ((1, 1, 32, 5120), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- ROW_MAJOR gamma: staging CB + compute-side tilize (other datapath) --
    "grid_starved": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, ttnn.bfloat16, RM, "loose"),
    # ---- ROW_MAJOR input: tilize / untilize path ----------------------------
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, RM, ttnn.bfloat16, RM, "loose"),
    # DISAMBIGUATOR (not in the op bench): ROW_MAJOR input with TILE gamma.
    # `row_major` moves BOTH knobs at once, so it cannot say whether an arm's
    # behaviour there belongs to the RM INPUT path or the RM GAMMA path.
    "row_major_gtile": ((1, 1, 8192, 1024), ttnn.bfloat16, RM, ttnn.bfloat16, TILE, "loose"),
    # ---- smallest supported cell: per-core-overhead regime ------------------
    "smallest": ((32, 17), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- non-tile-aligned W (Regime B, masked) and H (phantom-row clamp) -----
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    "h_nonalign": ((1, 1, 100, 736), ttnn.bfloat16, TILE, ttnn.bfloat16, TILE, "loose"),
    # ---- no gamma at all: the prologue is scaler + input only ----------------
    "no_gamma": ((1, 1, 32, 7168), ttnn.bfloat16, TILE, None, None, "loose"),
}

CONFIGS = {"default": default_compute_kernel_config, "loose": loose_compute_kernel_config}

# arm id -> levers.  `baseline` is the op's CURRENT prologue order, verbatim.
ARMS = {
    "baseline": {},  # scaler -> gamma -> input
    "reorder": {"prologue": 1},  # input -> gamma -> scaler
    "onebarrier": {"prologue": 2},  # issue all, ONE barrier, push in order
    "defer_gamma": {"prologue": 3},  # input first; gamma after the combine
    "wr_gamma": {"prologue": 4},  # gamma ingest moved to the writer (BRISC/NOC1)
    "defer_all": {"prologue": 5},  # arm 3 + the scaler built in the input's shadow
}


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

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item() / denom)


def plan_of(device, name):
    x, g, _, _ = make_tensors(device, name)
    cfg = CONFIGS[SHAPES[name][5]]()
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, x.memory_config())
    return blocking_plan(x, g, out, device, cfg, None)


def _dispatch(device, fn, iters=N_ITERS):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def run_arm(device, manifest, name, arm, iters=N_ITERS, extra_levers=None):
    x, g, _, _ = make_tensors(device, name)
    cfg = CONFIGS[SHAPES[name][5]]()
    levers = dict(ARMS[arm])
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
    print(f"{'shape':<16} {'arm':<14} {'ns':>10} {'vs baseline':>12}")
    for shape, arms in by_shape.items():
        base = arms.get("baseline")
        for arm, ns in arms.items():
            rel = f"{base / ns:.3f}x" if (base and ns) else ""
            print(f"{shape:<16} {arm:<14} {ns:>10.0f} {rel:>12}")
    return r
