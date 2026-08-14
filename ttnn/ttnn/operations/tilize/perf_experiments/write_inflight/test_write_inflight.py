# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""write_inflight END-TO-END arm — sub-page write transactions, correctness-gated.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/write_inflight/test_write_inflight.py

The roofline probe (`test_probe_writefloor.py`) closed every issue-side lever as
flat or worse, with ONE exception: on the fp32 output tile page (4096 B) the
same bytes issued as 2 or 4 sub-page transactions ran ~6-8% faster than one
whole-page transaction, while on the bf16 page (2048 B) the split was flat.

That is the OPPOSITE polarity to master.md B5 (`page_write`), which was measured
on the bf16 shapes and concluded "one whole tile page per transaction". This file
asks the only question that matters: does the probe's page-size effect move the
WALL of a real tilize whose writer is not the critical path?

`page_write=0` is the op's OWN already-implemented off-arm (two half-page
transactions per tile page), so this arm needs no new kernel and no new
correctness surface — it is a knob flip. Both arms are compared BIT-EXACT.

MEASURED — Wormhole n150 (bgd-lab-16). Every arm bit-exact vs the shipped
writer. Three PAIRED sessions (base and candidate back to back inside one
session, so the ratio is not exposed to the ~5% session-to-session drift of the
absolute numbers). ratio > 1 = the split is faster.

  case                base (median)  split2 (median)   ratios (3 sessions)
  widening_pad fp32       147,280        143,652       1.025 1.032 1.038
  a_square     fp32       177,559        181,889       1.017 0.961 0.976
  a_square     bf16        87,143         88,772       1.018 0.976 0.966
  b_wide_short bf16        12,950         13,132       1.007 0.986 0.964
  c_multiblock bf16       171,643        173,162       0.988 0.978 1.001
  d_smallest   bf16         3,110          3,141       0.977 0.990 0.990

Only the padded widening cast is consistently positive, at ~3% — and
`test_attrib_subpage.py` shows that 3% does NOT come from the write (the
writer's total occupancy is unchanged; it moves in `compute_tilize`). Everything
else is flat to mildly negative. NOT recommended for graduation as a write
lever.
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import ttnn
from loguru import logger

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))

import _bench_tilize as B  # noqa: E402


def _correctness(device, case, levers):
    """Bit-exact vs the shipped writer (page_write=1), padded positions included."""

    import torch

    shape = case["shape"]
    dtype = case["dtype"]
    # ONE input for both arms — regenerating it per arm compares different data.
    torch_in = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)

    def run(lv):
        from ttnn.operations.tilize import tilize
        from ttnn.operations.tilize import tilize_program_descriptor as pd

        saved = dict(pd.LEVERS)
        pd.LEVERS.update(lv)
        try:
            tt_in = ttnn.from_torch(
                torch_in,
                dtype=dtype,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            call = dict(use_multicore=True, use_double_buffer=True)
            if case.get("out_dtype"):
                call["dtype"] = case["out_dtype"]
            call.update(case.get("pad") or {})
            return ttnn.to_torch(tilize(tt_in, **call))
        finally:
            pd.LEVERS.update(saved)

    import torch

    ref = run(dict(page_write=1))
    got = run(levers)
    assert torch.equal(ref, got), f"{case['label']}: candidate is NOT bit-exact vs the shipped writer"


CASES = {
    # THE focus case: padded widening cast, fp32 output tile page = 4096 B.
    "widening_pad": dict(
        label="widening_pad/[1,1,1024,2048]->[1,1,2048,2048]fp32",
        shape=B._OUT_FILL_SHAPE[0],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=B._OUT_FILL_SHAPE[1], pad_value=10.2),
    ),
    # fp32 page, NO pad — isolates the page size from the pad machinery.
    "a_square_fp32": dict(label="a_square/fp32", shape=B.SHAPES["a_square"], dtype=ttnn.float32),
    # bf16 page (2048 B) — the regime B5 was originally measured on.
    "a_square_bf16": dict(label="a_square/bf16", shape=B.SHAPES["a_square"], dtype=ttnn.bfloat16),
    "b_wide_short": dict(label="b_wide_short/bf16", shape=B.SHAPES["b_wide_short"], dtype=ttnn.bfloat16),
    "c_multiblock": dict(label="c_multiblock/bf16", shape=B.SHAPES["c_multiblock"], dtype=ttnn.bfloat16),
    # smallest regime — a deeper/finer write pipeline can LOSE on per-core overhead.
    "d_smallest": dict(label="d_smallest/bf16", shape=B.SHAPES["d_smallest"], dtype=ttnn.bfloat16),
}


def _bench(device, case, levers, tag):
    return B._measure(
        device,
        case["shape"],
        case["dtype"],
        levers=levers,
        out_dtype=case.get("out_dtype"),
        pad=case.get("pad"),
        label=f"{tag}/{case['label']}",
    )


@pytest.mark.parametrize("name", list(CASES))
def test_subpage_write(device, name):
    """baseline = shipped writer (one whole tile page); candidate = two halves."""
    case = CASES[name]
    _correctness(device, case, dict(page_write=0))
    base = _bench(device, case, dict(page_write=1), "base")
    cand = _bench(device, case, dict(page_write=0), "split2")
    logger.info(f"WRITE_INFLIGHT {case['label']}: base={base:.0f} split2={cand:.0f} ns  ratio={base / cand:.3f}x")
