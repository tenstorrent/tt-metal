# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""read_inflight_v2 — the L1 / blocking interaction of a deeper input CB.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/read_inflight_v2/test_l1tight.py

``derive_blocking()`` reads ``cb_depth`` through ``wt_cap()``, so raising the
input CB from 2 to 3 groups SHRINKS the L1 ceiling on WT_CHUNK by a third — and
on a cell where that ceiling binds, the op would answer by splitting W finer,
which HALVES the read transfer. This file prices that trade on such a cell.

THE CELL: ``[1,1,2048,4096]`` bf16, interleaved DRAM -> interleaved DRAM, on
EIGHT cores (NT_H = 64 >> cores, so the pipeline floor is 1 block and the L1
ceiling is the only thing left to bind).

  depth 2 -> wt_cap = 1 MiB / (2 * (2048 + 2048)) = 128 = WT  -> n_chunks 1,
             WT_CHUNK 128, row_bytes 4096, 8 blocks/core, CBs = 1,048,576 B
             (EXACTLY the op's budget)
  depth 3 -> wt_cap = 85 < WT                                 -> n_chunks 2,
             WT_CHUNK 64, row_bytes 2048, 16 blocks/core, CBs = 655,360 B

So the three answers the coordinator can pick between are all measured here:
  (a) ship depth 2 at this blocking (cap the depth, ahead=1 with zero slack),
  (b) ship depth 3 and let derive_blocking() recompute the split,
  (c) ship depth 3 at the ORIGINAL blocking — priced by `test_depth3_unsplit`,
      which only records whether the allocation is physically possible.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
from loguru import logger

import ttnn

from ttnn.operations.tilize.perf_experiments.read_inflight_v2 import descriptor as D
from ttnn.operations.tilize.perf_experiments.read_inflight_v2.test_schedule import measure, run_arm

# NT_H = 64 tile-rows, WT = 128 tile-columns, 8 cores.
CELL = dict(kind="inter", shape=[1, 1, 2048, 4096], grid=(8, 1), n_chunks=1)

ARMS = [
    # (label, n_chunks, kwargs)
    ("split1/baseline_helper_d2", 1, dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2)),
    ("split1/trid_d2", 1, dict(variant=D.VARIANT_TRID, nt_blk=1, cb_depth=2)),
    ("split1/ahead1_d2", 1, dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=1)),
    # the control: is the FINER split good or bad on its own, with the op's own reader?
    ("split2/baseline_helper_d2", 2, dict(variant=D.VARIANT_HELPER, nt_blk=1, cb_depth=2)),
    ("split2/ahead1_d2", 2, dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=2, ahead=1)),
    # (b) depth 3 with derive_blocking() recomputed
    ("split2/ahead1_d3", 2, dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=3, ahead=1)),
    ("split2/ahead1_d4", 2, dict(variant=D.VARIANT_AHEAD, nt_blk=1, cb_depth=4, ahead=1)),
]


@pytest.mark.parametrize("label,n_chunks,kw", ARMS, ids=[a[0] for a in ARMS])
def test_l1tight(device, label, n_chunks, kw):
    measure(device, CELL, ttnn.bfloat16, f"L1TIGHT {label}", n_chunks=n_chunks, **kw)


def test_depth3_unsplit(device):
    """(c) depth 3 WITHOUT recomputing the split — over the op's CB budget.

    Records whether the allocation is even physically possible on this L1, and
    if it is, what it costs. Not a pass/fail on speed; a failure to allocate is
    itself the answer.
    """
    try:
        ns, exact, l1 = run_arm(
            device,
            CELL,
            ttnn.bfloat16,
            "L1TIGHT split1/ahead1_d3 (OVER BUDGET)",
            n_chunks=1,
            variant=D.VARIANT_AHEAD,
            nt_blk=1,
            cb_depth=3,
            ahead=1,
        )
        logger.info(f"L1TIGHT split1/ahead1_d3 ALLOCATED: ns={ns} exact={exact} cb_bytes={l1}")
        assert exact, "over-budget arm must still be bit-exact if it runs at all"
    except Exception as exc:  # allocation refusal is the finding
        logger.info(f"L1TIGHT split1/ahead1_d3 NOT ALLOCATABLE: {type(exc).__name__}: {str(exc)[:300]}")
