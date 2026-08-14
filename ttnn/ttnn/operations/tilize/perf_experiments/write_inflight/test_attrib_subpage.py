# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""write_inflight ATTRIBUTION — where does the widening pad's sub-page delta land?

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/write_inflight/test_attrib_subpage.py

`page_write=0` (two half-page transactions per tile page) measured a consistent
~3% paired win on the padded widening cast and flat-to-slightly-negative
everywhere else. Before that can be called a WRITE win at all, the delta has to
show up in `writer_issue`. If it does not, the 3% is pipeline reshuffling, not
bandwidth, and it should not be graduated as a write lever.

MEASURED — Wormhole n150 (bgd-lab-16), ns/core:

  stage                page_write=1      page_write=0     delta
  writer_issue              84,661            83,213      -1,448
  writer_wait                8,531             4,882      -3,649
  writer_stamp               3,591             3,580          -11
  writer_barrier               129             4,781      +4,652
  --- writer TOTAL (BRISC)  96,912            96,456        -456   (-0.5%)
  compute_tilize            91,299            81,561      -9,738   (-10.7%)
  reader TOTAL (NCRISC)     18,494            18,415         -79
  whole op                 143,348           140,936      -2,412

The writer's total occupancy is UNCHANGED (-0.5%): the split only moves time out
of `writer_wait`/`writer_issue` and into `writer_barrier`. The whole delta in the
wall tracks `compute_tilize` — the writer arriving later at the output CB leaves
compute less often blocked on `cb_reserve_back`. So the axis-I effect is a
pipeline phase shift, NOT write bandwidth, and the write side has no headroom.
"""

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

import pytest
import ttnn
from loguru import logger

sys.path.insert(0, os.path.join("tests", "ttnn", "unit_tests", "operations", "tilize"))

import _bench_tilize as B  # noqa: E402

from ttnn.operations.tilize.perf_experiments import _zones  # noqa: E402


@pytest.mark.parametrize("page_write", [1, 0], ids=["base", "split2"])
def test_attrib_widening_pad(device, page_write):
    shape, target = B._OUT_FILL_SHAPE
    _zones.clear()
    ns = B._measure(
        device,
        shape,
        ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=target, pad_value=10.2),
        levers=dict(page_write=page_write),
        label=f"attrib/page_write={page_write}",
    )
    stages, diag = _zones.breakdown()
    freq = device.get_clock_rate_mhz() if hasattr(device, "get_clock_rate_mhz") else 1000.0
    logger.info(f"\n=== ATTRIB page_write={page_write}  whole-op {ns:.0f} ns ===\n" + _zones.report(stages, diag, freq))
