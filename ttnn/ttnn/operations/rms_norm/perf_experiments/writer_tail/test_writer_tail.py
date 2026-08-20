# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED PERF BENCH — idea `writer_tail` (shorten the write tail).

The op's writer is STARVED for 6,983 of its 8,113 ns span on the focus case and
then pays `wr_issue` 814 + `wr_barrier` 423 = 1,237 ns entirely AFTER the last
compute — ~14% of the 9,050 ns wall, for SEVEN 2 KB one-packet page writes.
Three levers are measured here, alone and combined:

  pre    hoist every `TensorAccessor::get_noc_addr()` AHEAD of `cb_wait_front`,
         into the starvation window that is already being paid.
  sub    push the output in N-tile units so the writer's issue overlaps the tail
         of the multiply that is still producing it.
  share  hand N/16 of each block to the READER's idle NOC0.

`diag` is a DIAGNOSTIC arm, never shippable: it keeps the address generation and
drops the transfer, which is what decomposes `wr_issue` into "address
generation" vs "the write call" and decides which cost is worth attacking.

Two tests, kept apart on purpose:
  * test_wt_correctness — the GATE.  Every arm offered as an option must first
    reproduce torch to the focus case's soft PCC threshold (0.9995).
  * test_wt_bench       — the MEASUREMENT.  Dispatches a fixed, ordered set of
    arms and writes a manifest that `report()` folds the Tracy per-op CSV onto.

Run:
    scripts/run_safe_pytest.sh --dev <this file> -k correctness -s
    scripts/run_safe_pytest.sh --profile <this file> -k bench -s

Env:
    WT_MODE            focus | domain | all      WT_ARMS  comma list (repeats allowed)
    WT_DOMAIN_SHAPES   comma list                WT_SHARE_READY  enable the wt_share arms

===========================================================================
RESULT (Blackhole p150, 13x10 grid, 1350 MHz).  Every arm PCC-identical to
the baseline on all 10 shapes (0.99989-1.0, threshold 0.9995) - these arms
reorder issue, they do not change the math.
===========================================================================

WHICH COST THE TAIL ACTUALLY IS (focus, 7 pages of 2048 B per core):
    wr_issue 851 + wr_barrier 414 = 1,265 ns of a 9,096 ns wall.
    `diag_addr_only` (address generation kept, transfer dropped) puts
    wr_issue at 117 and wr_barrier at 27 - so ADDRESS GENERATION IS 117 ns
    and the `noc_async_write` payload is the other ~1,090 ns.
    458,752 output bytes / 1,090 ns = 421 GB/s aggregate, ~82% of this
    board's ~512 GB/s DRAM write peak.  The whole write therefore has
    <= ~190 ns (2.1% of the wall) of theoretical headroom.  That number,
    not any of the arms below, is the finding.

FOCUS (3 interleaved repeats per session, medians):
    baseline            9,012 / 9,153 / 9,096 / 9,213     (3 sessions)
    diag_addr_only      7,883 / 7,889          DIAGNOSTIC, not shippable
    pre  (hoist addrs)  9,259 / 9,204          0.97-0.99x  - the L1 round
                        trip through the scratch table costs more than the
                        117 ns of get_noc_addr it moves off the tail
    sub1 (fine push)    8,836 / 8,939 / 8,953 / 8,987      1.016-1.025x
    sub2 / sub4         flat
    pre_share8 (NOC0)   9,353                  0.979x     - see below

DOMAIN, `sub1`, 3 interleaved repeats each, medians:
    focus            9,096 ->  8,953   1.016x
    smallest         3,908 ->  3,906   1.001x
    prefill_7168   598,885 -> 600,377  0.998x
    row_major       91,601 ->  91,881  0.997x  (program is byte-identical
                                                here - pure noise, and the
                                                measured noise band, ~1.2%)
    prefill_1024_bf8 55,813 -> 56,396  0.990x
    decode_5120      8,230 ->   8,329  0.988x
    prefill_1024_f32 196,154 -> 198,523 0.988x
    prefill_1024    93,990 ->  95,297  0.986x
    decode_1024      5,747 ->   5,880  0.977x
    w_nonalign      20,344 ->  23,113  0.880x  <- material regression

WHY THE NOC SPLIT LOSES (pre_share8, reader takes 3 of 7 pages on NOC0):
    the writer`s own wr_issue went 851 ns for SEVEN pages -> 942 ns for
    FOUR (121 -> 236 ns/page).  Halving the pages per RISC made each page
    twice as expensive, which is the signature of a shared DOWNSTREAM
    bottleneck: the two NoCs feed the same DRAM controllers, so a second
    issuing port adds contention, not throughput.  Correct and hang-free on
    all 10 shapes (the barrier/ack discipline in wt_reader.cpp`s header
    holds), but 0.75-0.98x everywhere it is live.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import ttnn

from ttnn.operations.rms_norm.perf_experiments.writer_tail import wt_descriptor as wt

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MANIFEST_PATH = Path("generated/writer_tail_manifest.json")

N_WARMUP = 2
N_ITERS = 3

PCC_THRESHOLD = 0.9995


# ---------------------------------------------------------------------------
# The FROZEN precision contract of the focus case.  Never a perf lever: every arm
# in this file — baseline and candidate alike — runs under this identical
# descriptor.  bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False.
# ---------------------------------------------------------------------------
def loose_cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def default_cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


# name -> (shape, dtype, input_layout, gamma_layout, config)
# Mirrors `_bench_rms_norm.py`'s BENCH_SHAPES / BENCH_GAMMA_LAYOUT / GATESET.
SHAPES = {
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "decode_5120": ((1, 1, 32, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose"),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, "loose"),
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "prefill_1024_bf8": ((1, 1, 8192, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "loose"),
    "prefill_1024_f32": ((1, 1, 8192, 1024), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, "default"),
}

CONFIGS = {"loose": loose_cfg, "default": default_cfg}


def make(device, name):
    import torch

    shape, dtype, in_layout, gamma_layout, _ = SHAPES[name]
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=dtype, layout=in_layout, device=device)
    g = ttnn.from_torch(gt, dtype=dtype, layout=gamma_layout, device=device)
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
# Variants.  `baseline` is the op's CURRENT writer, verbatim (all knobs at 0).
# ---------------------------------------------------------------------------
VARIANTS = {
    "baseline": {},
    # DIAGNOSTIC ONLY (numerically wrong on purpose): address generation kept,
    # transfer dropped.  Prices the address-generation half of wr_issue.
    "diag_addr_only": dict(wt_diag=1),
    "pre": dict(wt_pre=1),
    "sub2": dict(wt_sub=2),
    "sub1": dict(wt_sub=1),
    "sub4": dict(wt_sub=4),
    "pre_sub2": dict(wt_pre=1, wt_sub=2),
    "pre_sub1": dict(wt_pre=1, wt_sub=1),
    "pre_sub4": dict(wt_pre=1, wt_sub=4),
    # Per-TILE output push at the solver's FULL DEST width: same fine writer
    # granularity as sub1, without shrinking the multiply's DEST block.
    "pertile": dict(wt_sub=1, wt_pertile=1),
}

# The NOC-split arm (wt_share) needs the reader side; added below only once that
# is implemented, so a half-built arm can never be dispatched into a hang.
# wt_share rides on the writer's precomputed address table (wt_pre), and does
# NOT compose with wt_sub — see the writer's `unit` comment.
SHARE_VARIANTS = {
    "pre_share4": dict(wt_pre=1, wt_share=4),
    "pre_share8": dict(wt_pre=1, wt_share=8),
}
if os.environ.get("WT_SHARE_READY"):
    VARIANTS.update(SHARE_VARIANTS)

# Arms that are numerically wrong BY CONSTRUCTION and are never offered as an
# option — they exist only to decompose a measured cost.
DIAGNOSTIC_ONLY = {"diag_addr_only"}


def run_once(device, name, variant, extra=None):
    x, g, _, _ = make(device, name)
    levers = dict(VARIANTS[variant])
    if extra:
        levers.update(extra)
    cfg = CONFIGS[SHAPES[name][4]]()
    return wt.wt_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=levers)


# ---------------------------------------------------------------------------
# Correctness gate
# ---------------------------------------------------------------------------
GATE_VARIANTS = [v for v in VARIANTS if v not in DIAGNOSTIC_ONLY]
GATE_SHAPES = list(SHAPES)


@pytest.mark.parametrize("variant", GATE_VARIANTS)
@pytest.mark.parametrize("name", GATE_SHAPES)
def test_wt_correctness(device, name, variant):
    import torch

    x, g, xt, gt = make(device, name)
    cfg = CONFIGS[SHAPES[name][4]]()
    out = ttnn.to_torch(wt.wt_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=dict(VARIANTS[variant])))
    ref = torch_ref(xt, gt)
    p = pcc(out, ref)
    print(f"\nWT_PCC {name}/{variant}: pcc={p:.6f} (threshold {PCC_THRESHOLD})")
    assert p >= PCC_THRESHOLD, f"{name}/{variant}: pcc {p} < {PCC_THRESHOLD}"


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


def arm(device, manifest, name, variant, iters=N_ITERS):
    x, g, _, _ = make(device, name)
    cfg = CONFIGS[SHAPES[name][4]]()
    levers = dict(VARIANTS[variant])
    n = _dispatch(device, lambda: wt.wt_rms_norm(x, gamma=g, compute_kernel_config=cfg, levers=levers), iters)
    label = f"{name}/{variant}"
    # A repeated arm gets a #k suffix: the bench is run 3x interleaved when a
    # call sits inside the ~2-3% device noise band (/perf-measure).
    k = sum(1 for a in manifest if a["label"].split("#")[0] == label)
    manifest.append(
        {
            "label": label if k == 0 else f"{label}#{k}",
            "shape": name,
            "variant": variant,
            "calls": n,
            "profiled": iters,
        }
    )


FOCUS_ARMS = [
    "baseline",
    "diag_addr_only",
    "pre",
    "sub1",
    "sub2",
    "sub4",
    "pre_sub1",
    "pre_sub2",
    "pre_sub4",
    "pertile",
    "pre_share4",
    "pre_share8",
]

DOMAIN_SHAPES = [
    "prefill_1024",
    "prefill_7168",
    "decode_1024",
    "decode_5120",
    "row_major",
    "smallest",
    "w_nonalign",
    "prefill_1024_bf8",
    "prefill_1024_f32",
]
DOMAIN_ARMS = ["baseline", "sub1", "pertile", "pre_share8"]


@pytest.mark.timeout(3600)
def test_wt_bench(device):
    mode = os.environ.get("WT_MODE", "all")
    only = os.environ.get("WT_ARMS")
    manifest = []

    if mode in ("focus", "all"):
        arms = only.split(",") if only else [a for a in FOCUS_ARMS if a in VARIANTS]
        for v in arms:
            arm(device, manifest, "focus", v)

    if mode in ("domain", "all"):
        sel = os.environ.get("WT_DOMAIN_SHAPES")
        shapes = [s for s in sel.split(",") if s] if sel else DOMAIN_SHAPES
        arms = only.split(",") if only else [a for a in DOMAIN_ARMS if a in VARIANTS]
        for name in shapes:
            for v in arms:
                arm(device, manifest, name, v)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWT_BENCH: manifest -> {MANIFEST_PATH} ({len(manifest)} arms)")
    assert manifest, "bench dispatched nothing"


def report(csv_path, manifest_path=MANIFEST_PATH):
    """Fold a per-op device CSV back onto the manifest labels, by dispatch order.

    Accepts either the Tracy `ops_perf_results_*.csv` (has an OP CODE column) or
    the raw `generated/profiler/.logs/cpp_device_perf_report.csv` the C++ side
    writes before Tracy's host/device stitching runs - the latter is the fallback
    when the stitching pass trips over a long dispatch train.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if rows and "OP CODE" in rows[0]:
        rows = [r for r in rows if r.get("OP CODE") == "GenericOpDeviceOperation"]
    # ALIGNMENT GUARD.  The fold is by dispatch ORDER, so a CSV that is short by
    # even one row silently shifts every later label onto the wrong arm (this bit
    # a long 20-arm run: it reported a 1.22x that a tightly interleaved re-run
    # showed was 0.995x).  Refuse to report a misaligned CSV.
    expected = sum(a["calls"] for a in manifest)
    assert len(rows) == expected, f"CSV has {len(rows)} op rows, manifest expects {expected} - fold would misalign"
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

    res = report(sys.argv[1])
    by_shape = {}
    for k, v in res.items():
        shape, variant = k.split("/", 1)
        by_shape.setdefault(shape, {})[variant] = v
    for shape, arms in by_shape.items():
        base = arms.get("baseline")
        print(f"\n== {shape}  (baseline {base})")
        for variant, v in arms.items():
            sp = f"{base / v:.3f}x" if (base and v) else ""
            print(f"   {variant:<20} {str(v):>12}  {sp}")
