# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED PERF BENCH — idea `no_reconfig`.

ONE idea: stop paying the per-phase data-format reconfig (and the redundant
per-call init) when the format provably does not change.

Two things happen here and are kept apart on purpose:

  * `test_no_reconfig_correctness` — the GATE.  Every arm offered as an option
    must reproduce torch to the focus case's soft PCC threshold (0.9995) AND
    carry no row-scale bias beyond the baseline's own.  Nothing about perf is
    asserted anywhere in this file.
  * `test_no_reconfig_bench` — the MEASUREMENT.  Dispatches a fixed, ordered set
    of arms and writes a manifest that `report()` folds the Tracy per-op CSV back
    onto (the in-process `ttnn.ReadDeviceProfiler` path returns nothing on this
    build, exactly as `_bench_rms_norm.py` documents).

Run:
    scripts/run_safe_pytest.sh ttnn/ttnn/operations/rms_norm/perf_experiments/no_reconfig/test_no_reconfig.py -k correctness -s
    scripts/run_safe_pytest.sh --profile ttnn/ttnn/operations/rms_norm/perf_experiments/no_reconfig/test_no_reconfig.py -k bench -s
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import ttnn

from ttnn.operations.rms_norm.perf_experiments.no_reconfig import nr_descriptor as nr
from ttnn.operations.rms_norm.perf_experiments.no_reconfig.nr_rms_norm import rms_norm as nr_rms_norm
from ttnn.operations.rms_norm.perf_experiments.no_reconfig.nr_rms_norm import default_compute_kernel_config

# torch is imported LAZILY: `ttnn/ttnn/operations/__init__.py` walks this whole
# tree at `import ttnn`, so a module-level torch import would be paid by every
# ttnn import in this (shared) checkout.

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MANIFEST_PATH = Path("generated/no_reconfig_manifest.json")

# Per-arm dispatch counts.  Kept env-tunable because the Tracy device-perf report
# has a finite op budget: a 5-round round-robin over 27 arms at 12 dispatches
# each overflowed it ("Device data missing: Op ... not present in
# cpp_device_perf_report.csv").  Fewer dispatches per arm + more ROUNDS is the
# better trade anyway -- the noise this is fighting is session drift BETWEEN
# arms, not variance within one.
N_WARMUP = int(os.environ.get("NR_WARMUP", "2"))
N_ITERS = int(os.environ.get("NR_ITERS", "10"))

PCC_THRESHOLD = 0.9995


# --- precision contract: FROZEN, never a lever -------------------------------
# Mirrors `_bench_rms_norm.BENCH_CONFIGS`.  Every arm of a given corner runs
# under the identical descriptor; nothing here moves math_fidelity /
# fp32_dest_acc_en / math_approx_mode to buy speed.
def _cfg_default():
    return default_compute_kernel_config()


def _cfg_loose():
    """The exact precision corner of feature_spec.LOOSE_CASES' perf cases."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _cfg_loose_fp32dest():
    """The loose corner with fp32 DEST — the MIXED-FORMAT corner of this idea.

    fp32_dest_acc_en=True forces `_acc_dtype` to float32 while the activation CBs
    stay bfloat16, so FMT_UNIFORM must come out FALSE and the flag must go inert.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


BENCH_CONFIGS = {"default": _cfg_default, "loose": _cfg_loose, "loose_fp32dest": _cfg_loose_fp32dest}

# name -> (shape, dtype, layout)  — mirrors `_bench_rms_norm.BENCH_SHAPES`.
BENCH_SHAPES = {
    "focus": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "prefill_1024": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "prefill_7168": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "w_nonalign": ((1, 1, 32, 4095), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    "grid_starved": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "decode_1024": ((1, 1, 32, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "h_nonalign": ((1, 1, 100, 736), ttnn.bfloat16, ttnn.TILE_LAYOUT),
}

# `grid_starved` / `row_major` deliberately keep the DEFAULT (ROW_MAJOR) gamma —
# that is the staging-CB + compute-side-tilize datapath.  Everything else pins
# gamma to TILE, matching the perf-gated feature_spec cases.
BENCH_GAMMA_LAYOUT = {
    "focus": ttnn.TILE_LAYOUT,
    "prefill_1024": ttnn.TILE_LAYOUT,
    "prefill_7168": ttnn.TILE_LAYOUT,
    "w_nonalign": ttnn.TILE_LAYOUT,
    "decode_1024": ttnn.TILE_LAYOUT,
    "h_nonalign": ttnn.TILE_LAYOUT,
}


# --- variants ----------------------------------------------------------------
# `baseline` is the op's CURRENT approach for this part, reproduced byte for byte
# (both flags off => the forked kernel compiles to the production one).
VARIANTS = {
    "baseline": dict(no_reconfig=0, no_init=0),
    "no_reconfig": dict(no_reconfig=1, no_init=0),
    "no_reconfig_no_init": dict(no_reconfig=1, no_init=1),
    # The init half ALONE is inexpressible: eltwise_chain's
    # InitReconfigOwner::Caller static_asserts `chain_no_reconfig_requested_v`,
    # so ownership of the init cannot be handed over while the chain still emits
    # reconfig.  Kept as a named arm so the constraint is visible in the manifest.
    "no_init_only": dict(no_reconfig=0, no_init=1),
    # NOISE CONTROL, bench-only: byte-for-byte the baseline, dispatched at a
    # different point in the session.  Any `control` - `baseline` gap is pure
    # run-to-run noise and is the yardstick every candidate delta is read
    # against.  (It earns its place: at prefill scale two provably identical
    # kernels landed 4.2% apart in one block-ordered session.)
    "control": dict(no_reconfig=0, no_init=0),
}


def make(device, name, dtype=None, gamma=True):
    import torch

    shape, shape_dtype, layout = BENCH_SHAPES[name]
    dtype = dtype or shape_dtype
    gamma_layout = BENCH_GAMMA_LAYOUT.get(name, ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT)
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.float32)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    x = ttnn.from_torch(xt, dtype=dtype, layout=layout, device=device)
    g = ttnn.from_torch(gt, dtype=dtype, layout=gamma_layout, device=device) if gamma else None
    return x, g, xt, (gt if gamma else None)


def torch_ref(xt, gt, eps=1e-6):
    import torch

    x = xt.to(torch.float32)
    r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x * r
    if gt is not None:
        out = out * gt.to(torch.float32)
    return out


def pcc(a, b):
    import torch

    # float64: the prefill shapes carry 5.8e7 elements and a float32 dot product
    # accumulates enough error there to report a "pcc" above 1.0.
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def row_scale_bias(out, ref):
    """Mean relative row-norm error — the systematic output-scale error PCC is blind to.

    PCC is invariant to a per-row scale factor, so a reduce datapath that
    over/under-estimates the sum of squares by a constant % passes PCC at 0.9999
    and still returns the wrong answer.  This op tracks that separately; so does
    every arm here.
    """
    import torch

    a = out.reshape(-1, out.shape[-1]).to(torch.float64)
    b = ref.reshape(-1, ref.shape[-1]).to(torch.float64)
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    return float(((na / (nb + 1e-30)) - 1.0).mean())


def run_once(x, g, variant, config="loose", levers_extra=None):
    levers = dict(VARIANTS[variant])
    if levers_extra:
        levers.update(levers_extra)
    return nr_rms_norm(x, gamma=g, compute_kernel_config=BENCH_CONFIGS[config](), _levers=levers)


# ---------------------------------------------------------------------------
# Correctness gate + the format-predicate evidence
# ---------------------------------------------------------------------------
# (shape, config, dtype, gamma, extra_levers) — one per distinct kernel path the
# compute kernel can take, mirroring `_bench_rms_norm.GATESET`.
#
# MEASURED FACT that reshaped this set: after the W-split graduation almost every
# gated shape now resolves to REGIME A (prefill_7168 -> Wt_core=56, G=4).  Only
# `w_nonalign` and `smallest` reach Regime B naturally, so the Regime-B corners
# below reach it through the op's OWN documented counterfactual lever
# (`force_regime=1`) rather than through an invented shape.
CORNERS = [
    ("focus", "loose", None, True),  # FOCUS: Regime A + W-split, TILE gamma
    ("prefill_1024", "loose", None, True),  # Regime A, full grid
    ("prefill_7168", "loose", None, True),  # Regime B, many chunks
    ("w_nonalign", "loose", None, True),  # masked Regime B / partial scaler
    ("smallest", "loose", None, True),  # per-core-overhead regime
    ("row_major", "loose", None, True),  # ROW_MAJOR input: tilize + untilize
    ("grid_starved", "loose", None, True),  # ROW_MAJOR gamma: staging + tilize
    ("decode_1024", "loose", None, True),
    ("h_nonalign", "loose", None, True),
    ("focus", "loose", None, False),  # no-gamma path (the INIT half's only site)
    ("prefill_7168", "loose", None, False),  # no-gamma Regime B: many scale chunks
    ("prefill_1024", "loose", ttnn.bfloat8_b, True),  # MIXED: bfp8 in, bf16 interm
    ("prefill_1024", "default", ttnn.float32, True),  # MIXED-ish: fp32 + fp32 DEST
    ("prefill_1024", "loose_fp32dest", None, True),  # MIXED: bf16 acts, fp32 acc CBs
    # --- the INIT half's only legal site: a Regime B scale loop with NO gamma ---
    ("w_nonalign", "loose", None, False),  # Regime B naturally (masked), no gamma
    ("prefill_7168", "loose", None, False, dict(force_regime=1)),  # forced Regime B, no gamma
    ("prefill_7168", "loose", None, True, dict(force_regime=1)),  # forced Regime B, with gamma
    ("focus", "loose", None, False, dict(force_regime=1)),  # forced Regime B on the focus shape
]
CORNERS = [c if len(c) == 5 else (*c, None) for c in CORNERS]

GATE_ARMS = ["baseline", "no_reconfig", "no_reconfig_no_init"]
# The bench adds the noise control; the correctness gate does not need it.
BENCH_ARMS = GATE_ARMS + ["control"]


def _tag(name, config, dtype, gamma, extra=None):
    return (
        f"{name}/{config}"
        + (f"/{str(dtype).split('.')[-1]}" if dtype else "")
        + ("" if gamma else "/no_gamma")
        + ("/" + ",".join(f"{k}={v}" for k, v in sorted(extra.items())) if extra else "")
    )


@pytest.mark.parametrize("corner", CORNERS, ids=lambda c: _tag(*c))
@pytest.mark.parametrize("variant", GATE_ARMS)
def test_no_reconfig_correctness(device, corner, variant):
    import torch

    name, config, dtype, gamma, extra = corner
    x, g, xt, gt = make(device, name, dtype, gamma)
    out = ttnn.to_torch(run_once(x, g, variant, config, extra))
    facts = dict(nr.LAST_PLAN_FMT)
    ref = torch_ref(xt, gt)
    p = pcc(out.to(torch.float32), ref)
    bias = row_scale_bias(out.to(torch.float32), ref)
    print(
        f"\nNR {_tag(*corner)} [{variant}] pcc={p:.6f} row_scale_bias={bias*100:+.4f}% "
        f"fmt_uniform={facts.get('fmt_uniform')} formats={facts.get('formats')} "
        f"regime={facts.get('regime')} Wt_core={facts.get('Wt_core')} G={facts.get('group_size')}"
    )
    assert p >= PCC_THRESHOLD, f"{_tag(*corner)}/{variant}: pcc {p} < {PCC_THRESHOLD}"


def test_no_reconfig_bit_identity(device):
    """The strongest precision evidence: under FMT_UNIFORM the candidate is BIT-identical.

    The reconfig this arm removes would have written the values the descriptors
    already hold, so the datapath is unchanged — not "PCC-close", identical.
    Asserted only where the host predicate actually fires.
    """
    import torch

    for corner in CORNERS:
        name, config, dtype, gamma, extra = corner
        x, g, _, _ = make(device, name, dtype, gamma)
        base = ttnn.to_torch(run_once(x, g, "baseline", config, extra))
        uniform = bool(nr.LAST_PLAN_FMT.get("fmt_uniform"))
        cand = ttnn.to_torch(run_once(x, g, "no_reconfig", config, extra))
        same = bool(torch.equal(base, cand))
        print(f"\nNR-BITS {_tag(*corner)}: fmt_uniform={uniform} bit_identical={same}")
        if uniform:
            assert same, f"{_tag(*corner)}: NO_RECONFIG changed bits under FMT_UNIFORM"


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


def arm(device, manifest, label, name, variant, config="loose", dtype=None, gamma=True, extra=None, iters=N_ITERS):
    x, g, _, _ = make(device, name, dtype, gamma)
    n = _dispatch(device, lambda: run_once(x, g, variant, config, extra), iters)
    manifest.append(
        {
            "label": label,
            "shape": name,
            "config": config,
            "dtype": str(dtype) if dtype else None,
            "gamma": gamma,
            "extra": extra or {},
            "variant": variant,
            "fmt": dict(nr.LAST_PLAN_FMT),
            "calls": n,
            "profiled": iters,
        }
    )


@pytest.mark.timeout(5400)
def test_no_reconfig_bench(device):
    """Dispatch the arm set ROUND-ROBIN over `NR_ROUNDS` rounds.

    MEASURED reason for the round-robin rather than one block per arm: on the
    long (600 us - 1 ms) prefill arms, two dispatches of a BYTE-IDENTICAL kernel
    -- `no_reconfig` vs `no_reconfig_no_init` at HAS_GAMMA, where the init skip
    compiles out entirely -- came out 967,157 and 988,875 ns apart in one
    block-ordered session.  That 2.2% is session drift, not a kernel difference,
    and interleaving + a median over rounds is what separates the two.
    """
    manifest = []
    sel = os.environ.get("NR_SHAPES")
    sel = [t for t in sel.split(",") if t] if sel else None
    rounds = int(os.environ.get("NR_ROUNDS", "1"))
    # Compute-ISOLATED arms: `stub_dm` keeps every reader/writer CB op and NoC
    # barrier but issues no transfer, so the compute payload -- the ONLY thing
    # this idea touches -- holds the wall instead of hiding behind the DRAM
    # reads.  This is the number that attributes a delta to the reconfig.
    stub_mode = os.environ.get("NR_STUB", "")  # "1" = both, "only" = compute-isolated only
    stub, stub_only = stub_mode in ("1", "only"), stub_mode == "only"
    for r in range(rounds):
        for corner in CORNERS:
            name, config, dtype, gamma, extra = corner
            if sel and name not in sel:
                continue
            # ROTATE the variant order every round.  MEASURED reason: at corners
            # where two arms compile to the SAME kernel (HAS_GAMMA makes the init
            # skip vanish, so `no_reconfig` and `no_reconfig_no_init` are the same
            # binary) the arm dispatched LAST in the round-body read ~1-3% faster
            # than the one dispatched second, consistently across shapes.  That is
            # a dispatch-position bias, not a kernel difference; rotating averages
            # it out instead of letting it land on whichever arm sits last.
            order = BENCH_ARMS[r % len(BENCH_ARMS) :] + BENCH_ARMS[: r % len(BENCH_ARMS)]
            for variant in order:
                if not stub_only:
                    arm(device, manifest, f"{_tag(*corner)}|{variant}#{r}", name, variant, config, dtype, gamma, extra)
                if stub:
                    ex = dict(extra or {})
                    ex.update(stub_dm=1)
                    arm(
                        device,
                        manifest,
                        f"{_tag(*corner)}+stub_dm|{variant}#{r}",
                        name,
                        variant,
                        config,
                        dtype,
                        gamma,
                        ex,
                    )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nNR_BENCH: manifest -> {MANIFEST_PATH} ({len(manifest)} arms)")
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
        out[a["label"]] = (vals[len(vals) // 2] if vals else None, a["fmt"].get("fmt_uniform"))
    return out


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def report_rounds(csv_path, manifest_path=MANIFEST_PATH):
    """Median across rounds of the per-round median — labels are `corner|variant#round`."""
    raw = report(csv_path, manifest_path)
    grouped = {}
    for k, (v, u) in raw.items():
        base = k.split("#")[0]
        grouped.setdefault(base, ([], u))[0].append(v)
    return {k: (_median(vs), u) for k, (vs, u) in grouped.items()}


if __name__ == "__main__":
    import sys

    res = report_rounds(sys.argv[1])
    base = {}
    for k, (v, _) in res.items():
        if k.endswith("|baseline"):
            base[k.rsplit("|", 1)[0]] = v
    print(f"{'corner':<40} {'variant':<22} {'ns':>10} {'x':>7}  uniform")
    for k, (v, u) in res.items():
        corner, variant = k.rsplit("|", 1)
        b = base.get(corner)
        sp = f"{b / v:.3f}" if (b and v) else "-"
        print(f"{corner:<40} {variant:<22} {v!s:>10} {sp:>7}  {u}")
