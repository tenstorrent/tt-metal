# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the group root's accumulation mechanism (`compute_root_sum`).

CORRECTNESS is the only pass/fail.  Every variant is gated against an fp64 torch
reference on the OP's own soft gates (PCC >= 0.9995, rel-RMS <= 0.04) applied to the
op-level quantity the stage feeds (`x * rsqrt(sum/W + eps)`), because that -- not the
raw sum -- is what the user sees.  The raw-sum error is reported too, unassessed.

Perf is MEASURED, never asserted.  Run:

  source python_env/bin/activate ; unset TT_METAL_DPRINT_CORES
  scripts/run_safe_pytest.sh --run-all --profile \
      ttnn/ttnn/operations/rms_norm/perf_experiments/root_sum_accumulate/test_root_sum_accumulate.py

then join the profiler CSV with this run's launch log (one JSONL line per launch, in
launch order):

  python3 ttnn/ttnn/operations/rms_norm/perf_experiments/root_sum_accumulate/report.py

Every launch is ONE `ttnn.generic_op`, so the CSV's rows and launches.jsonl's lines are
1:1 in order.  A variant that fails its gate still logs its line (the metrics are what
the coordinator needs), so the run uses --run-all and the gate is a reported flag rather
than an early exit -- except that a genuinely INCORRECT variant is also xfail-marked in
its own assertion at the end of the parametrized case.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("rsa_bench", Path(__file__).with_name("root_sum_accumulate.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

PCC_GATE = 0.9995
RELRMS_GATE = 0.04

LOG = Path(__file__).with_name("launches.jsonl")

# ITERS=1 keeps one launch == one real combine ROUND, so the measured fold is directly
# comparable to the op's `compute_root_sum` zone divided by its round count.  The `floor`
# variant prices launch + CB publish + drain, so the fold is a clean subtraction; no
# trial loop (device kernel time has no warm-up transient).
ITERS = int(os.environ.get("RSA_ITERS", "1"))
W_PER_CORE = int(os.environ.get("RSA_WPC", "128"))  # the focus shape's shard width, in elements

# The two LIVE geometries.
#   (8, 10)  (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8 -> 8 groups of 8, BLOCK_ROWS=10
#   (32, 1)  (1,1,32,5120)   WIDTH_SHARDED [32,160]   on 8x4 -> one group of 32, rows=1
FOCUS = [(8, 10), (32, 1)]

# Domain sweep.  GROUP_SIZE x rows, capped at 240 fp32 pages (~960 KB) of resident L1 --
# the same budget the op's own BLOCK_ROWS solve respects, so the excluded corners
# (e.g. GROUP_SIZE=32 x rows=32 = 4 MB) are not reachable op configurations.
# GROUP_SIZE=9 is a live geometry too (`wshard_w2304_9c`), and it is ODD on purpose.
_SWEEP_G = (4, 8, 9, 16, 28, 32)
_SWEEP_ROWS = (1, 2, 10, 32)
PAGE_CAP = 280  # fp32 pages; sized so the padded variant's extra slot does not shrink the set

SWEEP = [(g, r) for g in _SWEEP_G for r in _SWEEP_ROWS if bench.l1_pages(g, r) <= PAGE_CAP]

# Full menu on the focus geometries; the interesting subset across the sweep (keeps the
# JIT-compile count sane without narrowing the mechanism comparison).
SWEEP_VARIANTS = (
    "rmw",
    "pack_l1_acc",
    "pack_l1_acc_pairs",
    "dest_acc_wide",
    "dest_acc_wide_pad",
    "dest_acc_any",
    "dest_reuse_raw",
    "floor",
)


def _cases(geoms, variants):
    out = []
    for g, r in geoms:
        for v in variants:
            ok, _ = bench.is_expressible(v, g, r)
            if ok:
                out.append(pytest.param(v, g, r, id=f"{v}_g{g}_rows{r}"))
    return out


def _record(res, tag):
    line = dict(res)
    line["tag"] = tag
    if not res.get("ablation"):
        line["pcc_gate_met"] = res["pcc_out"] >= PCC_GATE
        line["relrms_gate_met"] = res["rel_rms_out"] <= RELRMS_GATE
    with LOG.open("a") as f:
        f.write(json.dumps(line) + "\n")
    print("LAUNCH " + json.dumps(line))
    return line


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    if LOG.exists():
        LOG.unlink()
    yield


@pytest.mark.parametrize("variant,group_size,rows", _cases(FOCUS, bench.VARIANTS))
def test_focus(device, variant, group_size, rows):
    res = bench.run_variant(device, variant, group_size, rows, w_per_core=W_PER_CORE, iters=ITERS)
    line = _record(res, "focus")
    if res.get("ablation"):
        pytest.skip("floor is a payload ablation: its output is undefined by construction")
    assert line["pcc_gate_met"] and line["relrms_gate_met"], (
        f"{variant} g={group_size} rows={rows}: pcc_out={res['pcc_out']:.6f} "
        f"rel_rms_out={res['rel_rms_out']:.5f} (sum rel-RMS {res['rel_rms_sum']:.2e}) "
        "-- INCORRECT / precision-trading at the op's soft gates"
    )


@pytest.mark.parametrize("variant,group_size,rows", _cases(SWEEP, SWEEP_VARIANTS))
def test_sweep(device, variant, group_size, rows):
    res = bench.run_variant(device, variant, group_size, rows, w_per_core=W_PER_CORE, iters=ITERS)
    line = _record(res, "sweep")
    if res.get("ablation"):
        pytest.skip("floor is a payload ablation: its output is undefined by construction")
    assert line["pcc_gate_met"] and line["relrms_gate_met"], (
        f"{variant} g={group_size} rows={rows}: pcc_out={res['pcc_out']:.6f} " f"rel_rms_out={res['rel_rms_out']:.5f}"
    )
