# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: does the combine path need the `compute_partial_handoff` copy?

CORRECTNESS is the only pass/fail (vs an fp64 torch reference, PCC >= 0.9995 and
rel-RMS <= 0.04 -- the op's soft gates).  Perf is MEASURED, never asserted:
run under `scripts/run_safe_pytest.sh --profile ...` and read
DEVICE KERNEL DURATION [ns] (column 20) out of
generated/profiler/reports/*/ops_perf_results_*.csv, one row per test in order.

Tests, in the order the profiler CSV will report them:

  test_focus[...]        the focus point (rows=10, width=1, acc_add) -- baseline
                         then candidate, back to back.
  test_sweep[...]        rows x width domain sweep.
  test_reduce_overhead[] the ADJACENT measurement the coordinator asked for: the
                         same reduce at 1 / 4 / 32 tiles of reduce-dim width per
                         call, candidate variant only, so per-call overhead can be
                         separated from reduction work.
  test_bitwise_identical the two variants' output tensors, compared on host.
"""

import importlib.util
from pathlib import Path

import pytest
import ttnn

# Path import (no package plumbing): this experiment dir is self-contained and must
# not depend on __init__.py files that parallel experiments also touch.
_spec = importlib.util.spec_from_file_location("rpth_bench", Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

# GATE CHOICE, stated explicitly.
#
# The op's soft gates (pcc >= 0.9995, rel-RMS <= 0.04) are declared on rms_norm's
# NORMALIZED OUTPUT.  This bench's output is the RAW sum(x^2) -- the quantity just
# before the rsqrt, which compresses relative error by ~2x and then divides it out.
# A raw bf16 sum of 32*width addends therefore has a legitimately lower PCC than the
# op it feeds (measured: pcc 0.99997 at width 1, 0.99899 at width 32), and a PCC gate
# on it would be a gate on the reduce datapath's intrinsic width scaling, not on the
# idea under test.
#
# So the HARD gates here are:
#   * rel-RMS <= 0.04                     -- the op's gate, dimensionally meaningful
#                                            on a raw sum, and passed everywhere.
#   * candidate output == baseline output  -- BITWISE.  This is the actual correctness
#                                            claim of the idea (see test_bitwise_identical).
# PCC is REPORTED for every point (it is what the coordinator's option menu needs) and
# gated only RELATIVELY: a variant may never be worse than the baseline it replaces.
RELRMS_GATE = 0.04
PCC_REPORT_FLOOR = 0.9995  # reported, not enforced -- see above

# The focus geometry, per rms_norm's (1,1,8192,1024) BLOCK_SHARDED shard [1024,128]:
#   32 tile-rows / core, BLOCK_ROWS = 10, WT_CHUNK = 4, X_SQUARED_WT = 1 (L6d fold),
#   NUM_W_CHUNKS = 1, AccumulateViaAdd.
FOCUS = dict(rows=10, width=1, blocks=4, algo="acc_add")


# rows x width sweep.  `blocks` keeps the TOTAL tile-rows near the focus shape's
# 32 per core so every point measures a comparable amount of work.
#   (rows*width) is capped at 320 tiles (640 KB of bf16) to stay inside L1 together
#   with the fp32 stat CBs -- the op is bounded the same way, so the excluded
#   (32,32) corner is not a reachable op configuration.
def _blocks_for(rows):
    return max(1, 32 // rows)


SWEEP = [
    pytest.param(rows, width, id=f"rows{rows}_w{width}")
    for rows in (1, 2, 10, 32)
    for width in (1, 4, 32)
    if rows * width <= 320
]


def _make_tensors(device, rows, width):
    """x^2-shaped bf16 reduce input + the fp32 stat output, both L1-sharded on core (0,0)."""
    import torch

    torch.manual_seed(1234)
    h, w = rows * bench.TILE, width * bench.TILE
    # x^2: non-negative, same magnitude distribution pass A actually produces.
    x = torch.randn((h, w), dtype=torch.float32)
    xsq = (x * x).to(torch.bfloat16)

    x_squared = ttnn.from_torch(
        xsq,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, w)),
    )
    stat_out = ttnn.from_torch(
        torch.zeros((h, bench.TILE), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bench.sharded_memory_config((h, bench.TILE)),
    )
    # fp64 reference: the REDUCE_ROW sum lands in column 0 of each stat tile.
    ref = xsq.to(torch.float64).sum(dim=-1)
    return x_squared, stat_out, ref


def _pcc_relrms(got, ref):
    import torch

    got = got.to(torch.float64)
    ref = ref.to(torch.float64)
    gc, rc = got - got.mean(), ref - ref.mean()
    denom = (gc.norm() * rc.norm()).item()
    pcc = 1.0 if denom == 0 else (gc * rc).sum().item() / denom
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    return pcc, rel_rms


def _run_and_check(device, variant, rows, width, blocks, algo):
    x_squared, stat_out, ref = _make_tensors(device, rows, width)
    out = bench.run(x_squared, stat_out, variant=variant, rows=rows, width=width, blocks=blocks, algo=algo)
    got = ttnn.to_torch(out)[:, 0]
    pcc, rel_rms = _pcc_relrms(got, ref)
    flag = "" if pcc >= PCC_REPORT_FLOOR else "  (below the op's 0.9995 output gate -- raw-sum PCC, see header)"
    print(
        f"\n[reduce_pack_to_handoff] variant={variant} rows={rows} width={width} "
        f"blocks={blocks} algo={algo}  pcc={pcc:.6f} rel_rms={rel_rms:.5f}{flag}"
    )
    assert rel_rms <= RELRMS_GATE, f"{variant} rows={rows} width={width}: rel_rms {rel_rms} > {RELRMS_GATE}"
    return got, pcc, rel_rms


# ---------------------------------------------------------------------------
# The focus point.  Baseline first, candidate second -- CSV rows 1 and 2.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", bench.VARIANTS)
def test_focus(device, variant):
    _run_and_check(device, variant, FOCUS["rows"], FOCUS["width"], FOCUS["blocks"], FOCUS["algo"])


# ---------------------------------------------------------------------------
# Domain sweep.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows, width", SWEEP)
@pytest.mark.parametrize("variant", bench.VARIANTS)
def test_sweep(device, variant, rows, width):
    _run_and_check(device, variant, rows, width, _blocks_for(rows), "acc_add")


# ---------------------------------------------------------------------------
# ADJACENT observation (measurement only): the reduce's own per-call cost vs its
# reduce-dim width.  Same number of reduce CALLS (rows*blocks = 30) at every
# width, so a flat curve means the call is overhead-bound, not work-bound.
# Both reduce datapaths, because the op picks between them.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("algo", bench.ALGOS)
@pytest.mark.parametrize("width", [1, 4, 32])
def test_reduce_overhead(device, algo, width):
    _run_and_check(device, "candidate", FOCUS["rows"], width, FOCUS["blocks"], algo)


# ---------------------------------------------------------------------------
# Is the candidate BYTE-IDENTICAL to the baseline?  (Not a perf test; this is the
# claim that lets the coordinator graduate it without a numerics discussion.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rows, width", SWEEP)
def test_bitwise_identical(device, rows, width):
    import torch

    blocks = _blocks_for(rows)
    b, b_pcc, _ = _run_and_check(device, "baseline", rows, width, blocks, "acc_add")
    c, c_pcc, _ = _run_and_check(device, "candidate", rows, width, blocks, "acc_add")
    assert torch.equal(b, c), f"rows={rows} width={width}: candidate differs from baseline"
    assert c_pcc >= b_pcc, f"rows={rows} width={width}: candidate pcc {c_pcc} < baseline {b_pcc}"
