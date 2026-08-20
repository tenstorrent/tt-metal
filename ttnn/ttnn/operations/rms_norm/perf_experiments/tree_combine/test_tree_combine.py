# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED PERF BENCH — idea `tree_combine` (hierarchical vs flat W-split combine).

Two things, kept apart on purpose:

  * `test_tc_correctness` — the GATE.  Every variant offered as an option must
    reproduce torch to the focus case's soft PCC threshold (0.9995) first.  A
    faster wrong answer is disqualified; nothing about perf is asserted here.
  * `test_tc_bench`       — the MEASUREMENT.  Dispatches a fixed, ordered set of
    arms and writes a manifest that `report()` folds the Tracy per-op CSV back
    onto (the in-process `ttnn.ReadDeviceProfiler` path returns nothing on this
    build, exactly as `_bench_rms_norm.py` documents).

Run:
  scripts/run_safe_pytest.sh ttnn/ttnn/operations/rms_norm/perf_experiments/tree_combine/test_tree_combine.py -k correctness -s
  scripts/run_safe_pytest.sh --profile ttnn/ttnn/operations/rms_norm/perf_experiments/tree_combine/test_tree_combine.py -k bench -s

Env:
  TC_MODE   focus (default) | domain | all
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import ttnn

from ttnn.operations.rms_norm.perf_experiments.tree_combine import tc_descriptor as tc

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MANIFEST_PATH = Path("generated/tc_manifest.json")

N_WARMUP = 2
N_ITERS = 10

PCC_THRESHOLD = 0.9995


def loose_cfg():
    """The EXACT precision contract of the focus case — frozen, never a perf lever.

    bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False.  Every arm in
    this file runs under this identical descriptor.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


# name -> (shape, gamma_layout[, input_layout])
SHAPES = {
    # THE focus case (feature_spec LOOSE_CASES perf case, gate <= 14_894 ns).
    "focus": ((1, 1, 32, 7168), ttnn.TILE_LAYOUT),
    # Prefill at the same width: many row-blocks per group, so the combine's
    # per-row-block price is paid Rt/BLOCK_HT times instead of once.
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.TILE_LAYOUT),
    # Narrow decode shapes that also W-split (Wt = 32 / 72 / 160).
    "decode_1024": ((1, 1, 32, 1024), ttnn.TILE_LAYOUT),
    "decode_2304": ((1, 1, 32, 2304), ttnn.TILE_LAYOUT),
    "decode_5120": ((1, 1, 32, 5120), ttnn.TILE_LAYOUT),
    # BLOCK_HT > 1 shapes (forced to G=32 in the arms below): the combine's
    # per-row-block handshake is then reused across several row-blocks AND each
    # gather carries BLOCK_HT tiles per core, which BLOCK_HT=1 never exercises.
    #   h1024_w7168 @ G=32 -> BLOCK_HT=8, 4 row-blocks / 2 groups
    #   prefill_1024 @ G=32 -> BLOCK_HT=8, 32 row-blocks / 2 groups
    "h1024_w7168": ((1, 1, 1024, 7168), ttnn.TILE_LAYOUT),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.TILE_LAYOUT),
    # The two OTHER kernel paths the combine has to keep correct.
    "rm_in_7168": ((1, 1, 1024, 7168), ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    "focus_rmgamma": ((1, 1, 32, 7168), ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
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

    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


# ---------------------------------------------------------------------------
# Variants — every arm is the SAME op, differing only in `combine_topology`
# (and, for the ceiling arms, `stub_combine`).  `flat` IS the op today.
# ---------------------------------------------------------------------------
def levers_for(topology, group=None, stub_combine=0, extra=None):
    lv = {"combine_topology": topology}
    if group:
        lv["w_group"] = group
    if stub_combine:
        lv["stub_combine"] = 1
    if extra:
        lv.update(extra)
    return lv


def run_once(x, g, topology, group=None, stub_combine=0, extra=None):
    return tc.tc_rms_norm(
        x, gamma=g, compute_kernel_config=loose_cfg(), levers=levers_for(topology, group, stub_combine, extra)
    )


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------

CORRECTNESS_CASES = [
    ("focus", "flat", 0),
    ("focus", "tree", 0),  # the plan's own G (32 -> 8x4)
    ("focus", "precollapse", 0),
    ("focus", "tree", 8),
    ("focus", "tree", 16),
    ("focus", "tree", 28),
    ("focus", "tree", 32),
    ("focus", "tree", 56),
    ("focus", "precollapse", 56),
    ("decode_1024", "tree", 0),
    ("decode_2304", "tree", 0),
    ("decode_5120", "tree", 0),
    ("prefill_7168", "tree", 0),
    ("rm_in_7168", "tree", 0),
    ("focus_rmgamma", "tree", 0),
    ("h1024_w7168", "tree", 32),
    ("h1024_w7168", "precollapse", 32),
    ("prefill_1024", "tree", 32),
]


@pytest.mark.parametrize("name,topology,group", CORRECTNESS_CASES, ids=lambda v: str(v))
def test_tc_correctness(device, name, topology, group):
    x, g, xt, gt = make(device, name)
    out = ttnn.to_torch(run_once(x, g, topology, group or None))
    ref = torch_ref(xt, gt)
    p = pcc(out, ref)
    print(f"\nTC_PCC {name}/{topology}/g{group or 'auto'}: pcc={p:.6f} (threshold {PCC_THRESHOLD})")
    assert p >= PCC_THRESHOLD, f"{name}/{topology}/g{group}: pcc {p} < {PCC_THRESHOLD}"


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


def arm(device, manifest, label, name, topology, group=None, stub_combine=0, extra=None, iters=N_ITERS):
    x, g, _, _ = make(device, name)
    n = _dispatch(device, lambda: run_once(x, g, topology, group, stub_combine, extra), iters)
    manifest.append(
        {
            "label": label,
            "shape": name,
            "topology": topology,
            "group": group or 0,
            "stub_combine": stub_combine,
            "extra": extra or {},
            "calls": n,
            "profiled": iters,
        }
    )


# Group sizes swept on the focus width (Wt = 224): each must divide Wt AND form a
# legal rectangle inside the grid (group_rect / _group_tiling).
FOCUS_GROUPS = [8, 16, 28, 32, 56]

DOMAIN = ["prefill_7168", "decode_1024", "decode_2304", "decode_5120", "rm_in_7168", "focus_rmgamma"]
# Shapes whose interesting arm is a FORCED G (their own plan picks a G whose
# rectangle is degenerate in y, where the tree has nothing to fold).
DOMAIN_FORCED = [("h1024_w7168", 32), ("prefill_1024", 32), ("focus", 56)]

STUB_BOTH = {"stub_dm": 1, "stub_compute": 1}


@pytest.mark.timeout(5400)
def test_tc_bench(device):
    mode = os.environ.get("TC_MODE", "focus")
    manifest = []

    if mode in ("focus", "all"):
        # 1. the ladder on the focus shape at the plan's own G (32).
        arm(device, manifest, "focus/flat", "focus", "flat")
        arm(device, manifest, "focus/tree", "focus", "tree")
        arm(device, manifest, "focus/precollapse", "focus", "precollapse")
        # 2. G sweep, both topologies.
        for gs in FOCUS_GROUPS:
            arm(device, manifest, f"focus/flat/g{gs}", "focus", "flat", gs)
            arm(device, manifest, f"focus/tree/g{gs}", "focus", "tree", gs)
        # 3. CEILING: the combine's compute payload removed, handshake intact.
        for gs in (32, 56):
            arm(device, manifest, f"focus/flat/g{gs}/stubcomb", "focus", "flat", gs, 1)
            arm(device, manifest, f"focus/tree/g{gs}/stubcomb", "focus", "tree", gs, 1)
        # 4. the combine FLOOR: all DRAM/eltwise payload stubbed too.
        for gs in (32, 56):
            arm(device, manifest, f"focus/flat/g{gs}/stub_both", "focus", "flat", gs, 0, STUB_BOTH)
            arm(device, manifest, f"focus/tree/g{gs}/stub_both", "focus", "tree", gs, 0, STUB_BOTH)

    if mode in ("domain", "all"):
        sel = os.environ.get("TC_DOMAIN_SHAPES")
        sel = [t for t in sel.split(",") if t] if sel else DOMAIN
        for name in sel:
            arm(device, manifest, f"{name}/flat", name, "flat")
            arm(device, manifest, f"{name}/tree", name, "tree")
            arm(device, manifest, f"{name}/precollapse", name, "precollapse")
        for name, gs in DOMAIN_FORCED:
            arm(device, manifest, f"{name}/flat/g{gs}", name, "flat", gs)
            arm(device, manifest, f"{name}/tree/g{gs}", name, "tree", gs)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nTC_BENCH: manifest -> {MANIFEST_PATH} ({len(manifest)} arms)")
    for a in manifest:
        print(f"  {a['label']:<32} shape={a['shape']:<14} topo={a['topology']:<12} g={a['group']}")
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
