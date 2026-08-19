# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm perf bench — device kernel duration + per-lever counterfactuals.

Every perf lever this op applies is a live knob in
`rms_norm_program_descriptor.LEVER_DEFAULTS`, threaded through the op's internal
`_levers=` hook.  This module measures the ON arm (all knobs at their applied
default) and the OFF arm (one knob flipped to its counterfactual) for each lever,
so re-measuring a lever on a new shape is one bench row instead of an
implement-measure-revert cycle.

Measurement is `DEVICE KERNEL DURATION [ns]` from the Tracy per-op CSV that
`scripts/run_safe_pytest.sh --profile` emits — the on-device number, never host
wall-clock.  (The in-process `ttnn.ReadDeviceProfiler` path returns nothing on
this build.)  Each arm runs a fixed number of dispatches in a deterministic
order; the bench writes a MANIFEST of `(label, n_calls)` alongside, and
`report_from_csv()` folds the CSV rows back onto the labels by position.

Bench shapes (op_design.md "Bench shapes", plus the ablation shape):

    grid_filling   (1, 1, 8192, 1024)   prefill, fills the grid, DRAM-bound
    wide_prefill   (1, 1, 8192, 7168)   prefill, widest hidden
    grid_starved   (1, 1, 32, 7168)     decode, Rt = 1 -> one core (Lamp L1 regime)
    smallest       (32, 17)             master.md B0 counterfactual regime
    row_major      (1, 1, 8192, 1024)   the RM/tilize path at the same size

Run it through the pytest wrapper so the device lock / reset / hang detection
apply:

    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_bench.py -s
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ttnn

from .rms_norm import default_compute_kernel_config, rms_norm

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 2
N_ITERS = 10

MANIFEST_PATH = Path("generated/rms_norm_bench_manifest.json")


# --- precision corners --------------------------------------------------------
# The cumulative bench set is measured at the op's OWN default precision corner
# ("default").  Refinement 1 added a second corner that matters: every
# feature_spec perf loose case runs at bf16 / HiFi2 / fp32_dest_acc_en=False, a
# different DEST width (8 tiles instead of 4) AND a different reduce datapath, so
# it has to be measured in its own right rather than inferred from the default.
def _cfg_default():
    return default_compute_kernel_config()


def _cfg_loose():
    """The exact precision corner of feature_spec.LOOSE_CASES' perf cases."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


BENCH_CONFIGS = {"default": _cfg_default, "loose": _cfg_loose}


# name -> (shape, dtype, layout)
BENCH_SHAPES = {
    "grid_filling": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "wide_prefill": ((1, 1, 8192, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "grid_starved": ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "smallest": ((32, 17), ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "row_major": ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
}


# lever id -> the `levers=dict(...)` counterfactual (OFF arm).  Each entry is
# written out as a `levers=dict(...)` forcing arm so `eval.verify_levers` can see
# that the counterfactual is re-runnable, not a one-off kernel edit.
def _arm(levers):
    return levers


LEVER_ARMS = {
    "A0": _arm(levers=dict(active_cores=32)),
    "A1": _arm(levers=dict(row_wise=0)),
    "B5": _arm(levers=dict(coalesce=0)),
    "B6": _arm(levers=dict(coalesce=0)),
    "B7": _arm(levers=dict(barrier_per_block=0)),
    "B9": _arm(levers=dict(noc_split=0)),
    "C16": _arm(levers=dict(double_buffer=0)),
    "compute_block_size": _arm(levers=dict(block_ht=1, dest_block=1)),
    "coarse_chunk": _arm(levers=dict(coarse_chunk=0)),
    # --- Refinement 1 -------------------------------------------------------
    # F-group: accumulator CBs forced back to unconditional fp32 (the Phase-0 arm).
    # Byte-identical at fp32_dest_acc_en=True, so measure it on the "loose" corner.
    "acc_narrow": _arm(levers=dict(acc_narrow=0)),
    # Regime B reduce datapath forced back to Phase 0's ReduceTile.  Only live on
    # the loose corner (see _reduce_via_add), so measure it there.
    "reduce_via_add": _arm(levers=dict(reduce_via_add=0)),
    # The W-chunk cap, now a first-class knob (block-size fidelity, finer than
    # coarse_chunk's all-or-nothing arm).
    "wt_block": _arm(levers=dict(wt_block=8)),
    # F25: the cheap DEST width is the ON arm, so the OFF arm is the caller's
    # fp32_dest_acc_en left alone (default corner) — see run_precision.
    "F25": _arm(levers=dict(dest_acc=0)),
    # F24: applied = the FAST bfloat8_b packer, so the OFF arm forces PRECISE.
    "F24": _arm(levers=dict(pack_precise=1)),
}

# Levers whose ON arm only exists at fp32_dest_acc_en=False, so measuring them on
# the default corner would compare two identical programs.
LOOSE_CORNER_LEVERS = ("acc_narrow", "reduce_via_add")
# Every lever run_levers must skip on the default corner: the loose-corner ones
# plus the two precision levers, which run_precision measures explicitly.
LOOSE_ONLY_LEVERS = LOOSE_CORNER_LEVERS + ("F24", "F25")

# /perf-measure ablation arms: the payload is stubbed, the CB/barrier sync
# scaffolding is intact, so the diff against the ON arm is that stage's cost.
ABLATION_ARMS = {
    "stub_dm": _arm(levers=dict(stub_dm=1)),
    "stub_compute": _arm(levers=dict(stub_compute=1)),
    "stub_both": _arm(levers=dict(stub_dm=1, stub_compute=1)),
}


def _dispatch(device, run_fn, iters=N_ITERS):
    """Warm up, then issue `iters` profiled dispatches.  Returns the profiled count."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    for _ in range(iters):
        run_fn()
    ttnn.synchronize_device(device)
    return N_WARMUP + iters


def _make(device, shape, dtype, layout):
    # torch is imported lazily: ttnn/ forbids a module-level torch import, and
    # this bench module lives beside the op.
    import torch

    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), dtype=dtype, layout=layout, device=device)
    # A block-float tensor must be TILE (ttnn refuses to build a ROW_MAJOR one),
    # so the gamma layout follows the dtype rather than being pinned.
    g = ttnn.from_torch(
        torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    return x, g


def run_arm(device, manifest, label, name, levers=None, iters=N_ITERS, config="default", dtype=None):
    """Dispatch one (shape, precision-corner, lever-setting) arm; append to the manifest.

    `config` names a BENCH_CONFIGS corner and `dtype` overrides the shape's dtype
    (Refinement 1: bfloat8_b), so a new precision point is a bench row rather than
    an edit to the shape table - the cumulative shape set stays the documented five.
    """
    shape, shape_dtype, layout = BENCH_SHAPES[name]
    dtype = dtype or shape_dtype
    x, g = _make(device, shape, dtype, layout)
    cfg = BENCH_CONFIGS[config]()
    n = _dispatch(device, lambda: rms_norm(x, gamma=g, compute_kernel_config=cfg, _levers=levers), iters)
    manifest.append(
        {
            "label": label,
            "shape": name,
            "config": config,
            "dtype": str(dtype),
            "levers": levers or {},
            "calls": n,
            "profiled": iters,
        }
    )


def run_baseline(device, names=None):
    """The cumulative, carried-forward bench set: every shape at the applied defaults."""
    manifest = []
    for n in names or list(BENCH_SHAPES):
        run_arm(device, manifest, f"baseline/{n}", n)
    return manifest


def run_levers(device, shape_name, levers=None):
    """ON/OFF pairs for every lever, on one bench shape."""
    manifest = []
    run_arm(device, manifest, f"{shape_name}/ON", shape_name)
    for lev in levers or [k for k in LEVER_ARMS if k not in LOOSE_ONLY_LEVERS]:
        run_arm(device, manifest, f"{shape_name}/OFF:{lev}", shape_name, LEVER_ARMS[lev])
    return manifest


def run_precision(device, shape_name):
    """Refinement 1: the two precision corners, the bf8b dtype, and the new levers.

    Every row is an ON/OFF pair on the corner where the lever is actually live, so
    `eval.verify_levers` sees a re-runnable counterfactual rather than an argument.
    """
    manifest = []
    run_arm(device, manifest, f"{shape_name}/prec:default", shape_name, config="default")
    run_arm(device, manifest, f"{shape_name}/prec:loose", shape_name, config="loose")
    run_arm(device, manifest, f"{shape_name}/prec:bf8b", shape_name, dtype=ttnn.bfloat8_b)
    run_arm(device, manifest, f"{shape_name}/prec:bf8b_loose", shape_name, config="loose", dtype=ttnn.bfloat8_b)
    for lev in LOOSE_CORNER_LEVERS:
        run_arm(device, manifest, f"{shape_name}/loose/OFF:{lev}", shape_name, LEVER_ARMS[lev], config="loose")
    # F25 attributed on its own: the DEST width only, fidelity held at the
    # default corner (the `loose` corner moves math_fidelity too and would
    # conflate F25 with F27).
    run_arm(device, manifest, f"{shape_name}/F25:dest_off", shape_name, LEVER_ARMS["F25"])
    # F24: fast (applied) vs precise bfloat8_b packer, on a bfloat8_b output.
    run_arm(device, manifest, f"{shape_name}/F24:bf8b_precise", shape_name, LEVER_ARMS["F24"], dtype=ttnn.bfloat8_b)
    return manifest


# Levers closed measured-no-payoff whose PREMISE moved in refinement-1 (new dtype,
# new reduce datapath, new derived CB formats).  verify_levers flags them
# "possibly unlocked"; this re-runs their arms on the regimes that actually
# changed, so the verdict is re-measured rather than re-argued.
RECHECK_LEVERS = ("A1", "B6", "C16")


def run_unlocked_recheck(device, shape_name):
    """ON/OFF for every 'possibly unlocked' lever, on each NEW regime."""
    manifest = []
    for tag, kw in (("loose", dict(config="loose")), ("bf8b", dict(dtype=ttnn.bfloat8_b))):
        run_arm(device, manifest, f"{shape_name}/{tag}/ON", shape_name, **kw)
        for lev in RECHECK_LEVERS:
            run_arm(device, manifest, f"{shape_name}/{tag}/OFF:{lev}", shape_name, LEVER_ARMS[lev], **kw)
    return manifest


def run_core_sweep(device, shape_name, caps=(16, 32, 64, 96, 0)):
    """Lever A0: the active-core count is a per-regime choice, so sweep it."""
    manifest = []
    for cap in caps:
        run_arm(
            device, manifest, f"{shape_name}/cores:{cap or 'full'}", shape_name, _arm(levers=dict(active_cores=cap))
        )
    return manifest


def run_gamma_replication(device, shape_name):
    """Quantify lever B12 / Lamp L2 headroom: gamma is re-read by EVERY core."""
    shape, dtype, layout = BENCH_SHAPES[shape_name]
    cfg = default_compute_kernel_config()
    manifest = []
    x, g = _make(device, shape, dtype, layout)
    n = _dispatch(device, lambda: rms_norm(x, gamma=g, compute_kernel_config=cfg))
    manifest.append(
        {"label": f"{shape_name}/with_gamma", "shape": shape_name, "levers": {}, "calls": n, "profiled": N_ITERS}
    )
    n = _dispatch(device, lambda: rms_norm(x, compute_kernel_config=cfg))
    manifest.append(
        {
            "label": f"{shape_name}/no_gamma",
            "shape": shape_name,
            "levers": {"gamma": "absent"},
            "calls": n,
            "profiled": N_ITERS,
        }
    )
    return manifest


def run_fidelity(device, shape_name):
    """Lever F27: the op's OWN default math_fidelity, measured against the ladder.

    F23's boundary: this arm varies the value the op would SHIP as its default,
    never a value a caller supplied.
    """
    shape, dtype, layout = BENCH_SHAPES[shape_name]
    x, g = _make(device, shape, dtype, layout)
    manifest = []
    for fid in ("HiFi4", "HiFi2", "LoFi"):
        cfg = default_compute_kernel_config()
        cfg.math_fidelity = getattr(ttnn.MathFidelity, fid)
        n = _dispatch(device, lambda c=cfg: rms_norm(x, gamma=g, compute_kernel_config=c))
        manifest.append(
            {
                "label": f"{shape_name}/fidelity:{fid}",
                "shape": shape_name,
                "levers": {"math_fidelity": fid},
                "calls": n,
                "profiled": N_ITERS,
            }
        )
    return manifest


def run_ablation(device, shape_name):
    """Bound classification: full run, then each payload stubbed, then BOTH at once.

    `stub_both` is the all-payloads-stubbed floor - the only run that licenses a
    statement about the whole op rather than about one balanced stage.
    """
    manifest = []
    run_arm(device, manifest, f"{shape_name}/ON", shape_name)
    for lev, arm in ABLATION_ARMS.items():
        run_arm(device, manifest, f"{shape_name}/ABL:{lev}", shape_name, arm)
    return manifest


def write_manifest(manifest, path=MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def report_from_csv(csv_path, manifest_path=MANIFEST_PATH):
    """Fold the Tracy per-op CSV back onto the manifest labels, by dispatch order."""
    manifest = json.loads(Path(manifest_path).read_text())
    with open(csv_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("OP CODE") == "GenericOpDeviceOperation"]
    out, i = {}, 0
    for arm in manifest:
        i += arm["calls"] - arm["profiled"]  # skip the warm-up dispatches
        window = rows[i : i + arm["profiled"]]
        i += arm["profiled"]
        vals = sorted(float(r[_DURATION_KEY]) for r in window if r.get(_DURATION_KEY))
        out[arm["label"]] = (vals[len(vals) // 2] if vals else None, arm["shape"], arm["levers"])
    return out
