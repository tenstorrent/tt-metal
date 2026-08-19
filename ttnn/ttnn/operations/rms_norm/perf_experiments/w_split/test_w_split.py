# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED PERF BENCH — idea `w_split` (cross-core split of the DEPENDENT W axis).

Two things happen here and they are kept apart on purpose:

  * `test_w_split_correctness`  — the GATE.  Every variant that is offered as an
    option must first reproduce torch to the focus case's soft PCC threshold
    (0.9995).  A faster wrong answer is disqualified; nothing about perf is
    asserted anywhere in this file.
  * `test_w_split_bench`        — the MEASUREMENT.  Dispatches a fixed, ordered
    set of arms and writes a manifest that `report_from_csv()` folds the Tracy
    per-op CSV back onto (the in-process `ttnn.ReadDeviceProfiler` path returns
    nothing on this build, exactly as `_bench_rms_norm.py` documents).

Run:
    scripts/run_safe_pytest.sh ttnn/ttnn/operations/rms_norm/perf_experiments/w_split/test_w_split.py -k correctness -s
    scripts/run_safe_pytest.sh --profile ttnn/ttnn/operations/rms_norm/perf_experiments/w_split/test_w_split.py -k bench -s
    python3 -c "import importlib.util as u; s=u.spec_from_file_location('t','<this file>'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.report('<csv>'))"

Env:
    WS_MODE   focus (default) | sweep | domain | all
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import ttnn

from ttnn.operations.rms_norm.perf_experiments.w_split import ws_descriptor as ws

# torch is imported LAZILY: `ttnn/ttnn/operations/__init__.py` walks this whole
# tree at `import ttnn`, so a module-level torch import here would be paid by every
# ttnn import in the checkout (and this checkout is shared with sibling
# experiments).

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MANIFEST_PATH = Path("generated/ws_split_manifest.json")

N_WARMUP = 2
N_ITERS = 10

PCC_THRESHOLD = 0.9995


def loose_cfg():
    """The EXACT precision contract of the focus case — frozen, never a perf lever.

    feature_spec LOOSE_CASES perf case: bf16 / HiFi2 / fp32_dest_acc_en=False /
    math_approx_mode=False.  Every arm in this file runs under this identical
    descriptor; nothing here ever moves one of these fields to buy speed.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


# name -> (shape, gamma_layout, input_layout)
SHAPES = {
    # THE focus case (feature_spec LOOSE_CASES perf case, goal <= 14_894 ns).
    "focus_7168": ((1, 1, 32, 7168), ttnn.TILE_LAYOUT),
    # Narrow decode: Wt=32, where the combine's fixed price may dominate.
    "decode_1024": ((1, 1, 32, 1024), ttnn.TILE_LAYOUT),
    # Very wide decode: Wt=1024.
    "decode_32768": ((1, 1, 32, 32768), ttnn.TILE_LAYOUT),
    # Rt=2: the shape where a plan has to CHOOSE row-parallel vs W-split.
    "rt2_12288": ((1, 1, 64, 12288), ttnn.TILE_LAYOUT),
    # Row-parallel-rich prefill, WIDE: the baseline already fills the grid, but per
    # core it is still Regime B (two DRAM reads of x).
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.TILE_LAYOUT),
    # The regime with NOTHING left for the W split to buy: the baseline already
    # fills the grid AND is already Regime A per core (Wt=32).  This is where a
    # measured regression, if there is one, has to show up.
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT),
    # The two OTHER kernel paths the W split has to keep correct: a ROW_MAJOR
    # input (reader stick reads + compute-side tilize/untilize, both now offset by
    # this core's column base) and a ROW_MAJOR gamma (staging CB + boot tilize).
    "rm_in_1024": ((1, 1, 8192, 1024), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "focus_rmgamma": ((1, 1, 32, 7168), ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
    # A WIDE ROW_MAJOR input: the per-core stick slice stays large enough that the
    # stick read may still be NoC-efficient.  This is what decides whether the RM
    # regression is "the RM path" or "a too-small per-core stick".
    "rm_in_7168": ((1, 1, 1024, 7168), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "rm_in_1024_narrow_rt": ((1, 1, 1024, 1024), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
}


def make(device, name):
    import torch

    entry = SHAPES[name]
    shape, gamma_layout = entry[0], entry[1]
    in_layout = entry[2] if len(entry) > 2 else ttnn.TILE_LAYOUT
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=in_layout, device=device)
    g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=gamma_layout, device=device)
    return x, g, xt, gt


def torch_ref(xt, gt, eps=1e-6):
    import torch

    x = xt.to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * gt.to(torch.float32)


def pcc(a, b):
    import torch

    # float64: the prefill shapes carry 5.8e7 elements and a float32 dot product
    # accumulates enough error there to report a "pcc" above 1.0.
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
# Each entry is the kwargs handed to ws.ws_rms_norm.  `baseline` is the op's
# CURRENT approach for this part (row-parallel only, no W split).
def variant_kwargs(variant, group_size):
    if variant == "baseline":
        return dict(group_size=0)
    if variant == "mcast":
        return dict(group_size=group_size, topology="mcast")
    if variant == "unicast":
        return dict(group_size=group_size, topology="unicast")
    if variant == "no_combine":
        # W split with the cross-core combine REMOVED (each core finalizes its own
        # slice).  Numerically WRONG on purpose: it prices the combine against the
        # pure-parallelism ceiling and is never offered as a shippable option.
        return dict(group_size=group_size, topology="mcast", combine_stub=1)
    raise ValueError(variant)


def run_once(x, g, variant, group_size, levers=None):
    return ws.ws_rms_norm(
        x, gamma=g, compute_kernel_config=loose_cfg(), levers=levers, **variant_kwargs(variant, group_size)
    )


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------

CORRECTNESS_CASES = [
    ("focus_7168", "baseline", 0),
    ("focus_7168", "mcast", 4),
    ("focus_7168", "mcast", 8),
    ("focus_7168", "mcast", 14),
    ("focus_7168", "mcast", 16),
    ("focus_7168", "mcast", 28),
    ("focus_7168", "mcast", 32),
    ("focus_7168", "mcast", 56),
    ("focus_7168", "unicast", 56),
    ("decode_1024", "mcast", 8),
    ("decode_1024", "mcast", 32),
    ("decode_32768", "mcast", 32),
    ("decode_32768", "mcast", 64),
    ("rt2_12288", "mcast", 32),
    ("prefill_7168", "mcast", 4),
    ("prefill_7168", "mcast", 8),
    ("prefill_1024", "mcast", 4),
    ("rm_in_1024", "mcast", 4),
    ("focus_rmgamma", "mcast", 32),
]


@pytest.mark.parametrize("name,variant,group", CORRECTNESS_CASES, ids=lambda v: str(v))
def test_w_split_correctness(device, name, variant, group):
    x, g, xt, gt = make(device, name)
    out = ttnn.to_torch(run_once(x, g, variant, group))
    ref = torch_ref(xt, gt)
    p = pcc(out, ref)
    print(f"\nWS_PCC {name}/{variant}/g{group}: pcc={p:.6f} (threshold {PCC_THRESHOLD})")
    assert p >= PCC_THRESHOLD, f"{name}/{variant}/g{group}: pcc {p} < {PCC_THRESHOLD}"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _dispatch(device, fn, iters=N_ITERS):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def arm(device, manifest, label, name, variant, group, levers=None, iters=N_ITERS):
    x, g, _, _ = make(device, name)
    n = _dispatch(device, lambda: run_once(x, g, variant, group, levers), iters)
    manifest.append(
        {
            "label": label,
            "shape": name,
            "variant": variant,
            "group": group,
            "levers": levers or {},
            "calls": n,
            "profiled": iters,
        }
    )


# group sizes swept on the focus width (Wt=224); each must DIVIDE Wt and fit the
# grid as a rectangle.
FOCUS_GROUPS = [4, 8, 14, 16, 28, 32, 56]

# (shape, group sizes) for the domain sweep.
DOMAIN = [
    ("decode_1024", [4, 8, 16, 32]),  # Wt=32, Rt=1 (baseline already Regime A)
    ("decode_32768", [16, 32, 64]),  # Wt=1024, Rt=1
    ("rt2_12288", [8, 16, 32, 64]),  # Rt=2 -> row-parallel vs W-split crossover
    ("prefill_7168", [2, 4, 8, 16]),  # Rt=256, grid-filling, per-core Regime B
    ("prefill_1024", [2, 4, 8, 16, 32]),  # Rt=256, grid-filling, per-core Regime A already
    ("rm_in_1024", [2, 4, 8]),  # ROW_MAJOR input path (tilize/untilize)
    ("rm_in_1024_narrow_rt", [2, 4, 8]),
    ("rm_in_7168", [8, 16, 32]),
    ("focus_rmgamma", [32]),  # ROW_MAJOR gamma path (staging CB + boot tilize)
]


@pytest.mark.timeout(3600)
def test_w_split_bench(device):
    mode = os.environ.get("WS_MODE", "all")
    manifest = []

    if mode in ("focus", "all"):
        arm(device, manifest, "focus/baseline", "focus_7168", "baseline", 0)
        for gs in FOCUS_GROUPS:
            arm(device, manifest, f"focus/mcast/g{gs}", "focus_7168", "mcast", gs)
        arm(device, manifest, "focus/unicast/g56", "focus_7168", "unicast", 56)
        arm(device, manifest, "focus/unicast/g14", "focus_7168", "unicast", 14)
        # combine price: the same W split with the cross-core combine removed.
        for gs in (14, 28, 56):
            arm(device, manifest, f"focus/no_combine/g{gs}", "focus_7168", "no_combine", gs)
        # combine FLOOR: payload stubbed (no DRAM x reads, no eltwise math), the
        # whole semaphore + mcast scaffolding intact.  Compared against the
        # baseline's own stubbed floor.
        stub = dict(stub_dm=1, stub_compute=1)
        arm(device, manifest, "focus/baseline/stub_both", "focus_7168", "baseline", 0, stub)
        arm(device, manifest, "focus/mcast/g56/stub_both", "focus_7168", "mcast", 56, stub)
        arm(device, manifest, "focus/mcast/g14/stub_both", "focus_7168", "mcast", 14, stub)
        # ... and the SAME stubbed arms with the combine removed: the difference is
        # the combine's own fixed price, with no payload behind it.
        arm(device, manifest, "focus/no_combine/g56/stub_both", "focus_7168", "no_combine", 56, stub)
        arm(device, manifest, "focus/no_combine/g14/stub_both", "focus_7168", "no_combine", 14, stub)

    if mode in ("domain", "all"):
        sel = os.environ.get("WS_DOMAIN_SHAPES")
        sel = [t for t in sel.split(",") if t] if sel else None
        for name, groups in DOMAIN:
            if sel and name not in sel:
                continue
            arm(device, manifest, f"{name}/baseline", name, "baseline", 0)
            for gs in groups:
                arm(device, manifest, f"{name}/mcast/g{gs}", name, "mcast", gs)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWS_BENCH: manifest -> {MANIFEST_PATH} ({len(manifest)} arms)")
    for a in manifest:
        print(f"  {a['label']:<34} shape={a['shape']:<14} variant={a['variant']:<10} g={a['group']}")
    assert manifest, "bench dispatched nothing"


def report(csv_path, manifest_path=MANIFEST_PATH):
    """Fold the Tracy per-op CSV back onto the manifest labels, by dispatch order."""
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = {}, 0
    for a in manifest:
        i += a["calls"] - a["profiled"]
        window = rows[i : i + a["profiled"]]
        i += a["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in window if r.get(_DURATION_KEY))
        out[a["label"]] = vals[len(vals) // 2] if vals else None
    return out


if __name__ == "__main__":
    import sys

    for k, v in report(sys.argv[1]).items():
        print(f"{k:<40} {v}")
