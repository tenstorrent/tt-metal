# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: where the group root's running sum LIVES (`compute_root_sum`).

BASELINE = `pack_l1_acc`, the op's CURRENT in-tree fold (D16).  CORRECTNESS is the only
pass/fail: every variant is gated against an fp64 torch reference on the op's own soft gates
(PCC >= 0.9995, rel-RMS <= 0.04) applied to the op-level quantity the stage feeds
(`x * rsqrt(sum/W + eps)`), because that -- not the raw sum -- is what the user sees.  The
raw-sum error is reported too, unassessed.

Perf is MEASURED, never asserted.  Run:

  source python_env/bin/activate ; unset TT_METAL_DPRINT_CORES
  scripts/run_safe_pytest.sh --run-all --profile \
      ttnn/ttnn/operations/rms_norm/perf_experiments/root_sum_dest_accumulate/test_root_sum_dest_accumulate.py

then join the profiler CSV with this run's launch log (one JSONL line per launch, in
launch order):

  python3 ttnn/ttnn/operations/rms_norm/perf_experiments/root_sum_dest_accumulate/report.py

Every launch is ONE `ttnn.generic_op`, so the CSV's rows and launches.jsonl's lines are 1:1
in order.  A variant that fails its gate still logs its line (the metrics are what the
coordinator needs), so the run uses --run-all and the gate is a reported flag.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("rsda_bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

PCC_GATE = 0.9995
RELRMS_GATE = 0.04

LOG = Path(__file__).with_name("launches.jsonl")

# ITERS=1 keeps one launch == one real combine ROUND, so the measured fold is directly
# comparable to the op's `compute_root_sum` zone divided by its round count.  The `floor`
# variant prices launch + CB publish + drain, so the fold is a clean subtraction; no trial
# loop (device kernel time has no warm-up transient).
ITERS = int(os.environ.get("RSDA_ITERS", "1"))
W_PER_CORE = int(os.environ.get("RSDA_WPC", "128"))  # the focus shape's shard width, in elements

# THE FOCUS GEOMETRY, and the op's real WIDTH-shard decode profiles.
#   (8, 8)   (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8 -> 8 groups of 8, BLOCK_ROWS=8
#   (32, 1)  (1,1,32,5120)   WIDTH_SHARDED on 8x4 -> one group of 32, rows=1
#   (28, 1)  (1,1,32,7168)   WIDTH_SHARDED, 28 cores
#   (9, 1)   (1,1,32,2304)   WIDTH_SHARDED, 9 cores -- ODD on purpose
#   (8, 1)   (1,1,32,1024)   WIDTH_SHARDED, 8 cores
FOCUS = [(8, 8), (32, 1), (28, 1), (9, 1), (8, 1)]

# Domain sweep: GROUP_SIZE x rows-per-block.  Capped at 300 fp32 pages (~1.2 MB) of resident
# L1 -- the same budget the op's own BLOCK_ROWS solve respects, so the excluded corners (e.g.
# GROUP_SIZE=8 x rows=32 = 385 pages = 1.54 MB > L1) are not reachable op configurations.
_SWEEP_G = (4, 8, 9, 16, 28, 32)
_SWEEP_ROWS = (1, 8, 32)
PAGE_CAP = 300

SWEEP = [(g, r) for g in _SWEEP_G for r in _SWEEP_ROWS if bench.l1_pages(g, r) <= PAGE_CAP]
INFEASIBLE = [(g, r) for g in _SWEEP_G for r in _SWEEP_ROWS if bench.l1_pages(g, r) > PAGE_CAP]


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
    print(f"# infeasible geometries (over the {PAGE_CAP}-page L1 cap, not op-reachable): {INFEASIBLE}")
    yield


def _run(device, variant, group_size, rows, tag):
    res = bench.run_variant(device, variant, group_size, rows, w_per_core=W_PER_CORE, iters=ITERS)
    line = _record(res, tag)
    if res.get("ablation"):
        pytest.skip("floor is a payload ablation: its output is undefined by construction")
    assert line["pcc_gate_met"] and line["relrms_gate_met"], (
        f"{variant} g={group_size} rows={rows}: pcc_out={res['pcc_out']:.6f} "
        f"rel_rms_out={res['rel_rms_out']:.5f} (sum rel-RMS {res['rel_rms_sum']:.2e}) "
        "-- INCORRECT / precision-trading at the op's soft gates"
    )


@pytest.mark.parametrize("variant,group_size,rows", _cases(FOCUS, bench.VARIANTS))
def test_focus(device, variant, group_size, rows):
    _run(device, variant, group_size, rows, "focus")


@pytest.mark.parametrize("variant,group_size,rows", _cases(SWEEP, bench.VARIANTS))
def test_sweep(device, variant, group_size, rows):
    _run(device, variant, group_size, rows, "sweep")
