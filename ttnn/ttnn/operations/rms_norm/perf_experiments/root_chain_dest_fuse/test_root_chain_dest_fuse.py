# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: FUSE the group root's `compute_root_sum` + `compute_root_finalize`.

CORRECTNESS is the only pass/fail.  Every variant is gated against an fp64 torch
reference on the OP's own soft gates (PCC >= 0.9995, rel-RMS <= 0.04) applied to the
op-level quantity the stage pair feeds (`x * rsqrt(sum/W + eps)`), because that -- not
the raw stat -- is what the user sees.  The stat's own error is reported too.

Perf is MEASURED, never asserted.  Run:

  source python_env/bin/activate ; unset TT_METAL_DPRINT_CORES
  scripts/run_safe_pytest.sh --run-all --profile \
      ttnn/ttnn/operations/rms_norm/perf_experiments/root_chain_dest_fuse/test_root_chain_dest_fuse.py

then join the profiler CSV with this run's launch log (one JSONL line per launch, in
launch order):

  python3 ttnn/ttnn/operations/rms_norm/perf_experiments/root_chain_dest_fuse/report.py

Every launch is ONE `ttnn.generic_op`, so the CSV's rows and launches.jsonl's lines are
1:1 in order.  A variant that fails its gate still logs its line (the metrics are what
the coordinator needs), so the run uses --run-all and the gate is a reported flag as well
as an assertion at the end of the parametrized case.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("rcdf_bench", Path(__file__).with_name("chain_bench.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

PCC_GATE = 0.9995
RELRMS_GATE = 0.04

LOG = Path(__file__).with_name("launches.jsonl")

# ITERS=1 keeps one launch == one real combine ROUND, so the measured stage pair is
# directly comparable to the op's (compute_root_sum + compute_root_finalize) zones divided
# by their round count.  `floor` prices launch + CB publish, so the stage is a clean
# subtraction; no trial loop (device kernel time has no warm-up transient).
ITERS = int(os.environ.get("RCDF_ITERS", "1"))
W_PER_CORE = int(os.environ.get("RCDF_WPC", "128"))  # the focus shape's shard width, in elements

# The LIVE geometries (full menu on each).
#   (8, 8)   (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8 -> 8 groups of 8, BLOCK_ROWS=8
#            == THE FOCUS SHAPE (4 combine rounds per core)
#   (32, 1)  (1,1,32,5120)   WIDTH_SHARDED [32,160]   on 8x4 -> one group of 32, rows=1
#   (28, 1)  (1,1,32,7168)   WIDTH_SHARDED            -> one group of 28, rows=1
FOCUS = [(8, 8), (32, 1), (28, 1)]

# Domain sweep: GROUP_SIZE x rows-per-block, capped at 288 fp32 pages (1.125 MB) of
# resident L1 -- the same budget the op's own BLOCK_ROWS solve respects, so the excluded
# corners are not reachable op configurations.  GROUP_SIZE=9 is a live geometry
# (`wshard_w2304_9c`) and is ODD on purpose.
_SWEEP_G = (4, 8, 9, 16, 28, 32)
_SWEEP_ROWS = (1, 8, 32)
PAGE_CAP = 288

SWEEP = [(g, r) for g in _SWEEP_G for r in _SWEEP_ROWS if bench.l1_pages(g, r) <= PAGE_CAP]

# The interesting subset across the sweep (keeps the JIT-compile count sane without
# narrowing the mechanism comparison): the op today, the fold-only swap, the two fused
# forms that differ in whether they need a padded gather slot, and the ablation.
SWEEP_VARIANTS = ("baseline", "destacc_split", "fused_pairs", "fused_pairs_stream", "fused_reuse", "floor")

# The op supports fp32_dest_acc_en=True as well; the focus case pins False and the perf
# menu is measured there ONLY (the precision contract is never a lever).  This slice
# exists purely to prove the fused form is CORRECT at the other setting, so that
# `fp32_dest_acc_en=True` cannot be mistaken for an untested exception.  Perf is still
# logged, but it is a different config and must not be compared across the two.
DEST_FP32_VARIANTS = ("baseline", "fused_pairs", "fused_pairs_stream", "fused_reuse")


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


def _run(device, variant, group_size, rows, tag, dest_fp32=False):
    res = bench.run_variant(device, variant, group_size, rows, w_per_core=W_PER_CORE, iters=ITERS, dest_fp32=dest_fp32)
    line = _record(res, tag)
    if res.get("ablation"):
        pytest.skip("floor is a payload ablation: its output is undefined by construction")
    assert line["pcc_gate_met"] and line["relrms_gate_met"], (
        f"{variant} g={group_size} rows={rows}: pcc_out={res['pcc_out']:.6f} "
        f"rel_rms_out={res['rel_rms_out']:.5f} (stat rel-RMS {res['rel_rms_stat']:.2e}) "
        "-- INCORRECT / precision-trading at the op's soft gates"
    )


@pytest.mark.parametrize("variant,group_size,rows", _cases(FOCUS, bench.VARIANTS))
def test_focus(device, variant, group_size, rows):
    _run(device, variant, group_size, rows, "focus")


@pytest.mark.parametrize("variant,group_size,rows", _cases(SWEEP, SWEEP_VARIANTS))
def test_sweep(device, variant, group_size, rows):
    _run(device, variant, group_size, rows, "sweep")


@pytest.mark.parametrize("variant,group_size,rows", _cases([(8, 8), (32, 1), (9, 8)], DEST_FP32_VARIANTS))
def test_dest_fp32(device, variant, group_size, rows):
    """CORRECTNESS ONLY, at the op's other fp32_dest_acc_en.  Never compared to the menu."""
    _run(device, variant, group_size, rows, "dest_fp32", dest_fp32=True)
