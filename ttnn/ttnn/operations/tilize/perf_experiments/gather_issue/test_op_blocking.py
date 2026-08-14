# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the COALESCE precondition costs the rest of the op.

The coalesced gather needs `row_bytes == page_bytes`, i.e. the host must pick
WT_CHUNK = the SOURCE shard's tile width (4 here) instead of the coarsest chunk
that fits L1 (8). That is a block-width change the compute and writer also see,
so the reader-only bake-off cannot price it — this does, on the REAL op, by
capping the W block factor from the outside (`wt_cap` monkeypatched in this test
process only; the op's own files are untouched).

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/gather_issue/test_op_blocking.py
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

from ttnn.operations.tilize import tilize_program_descriptor as pd  # noqa: E402

RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    logger.info(
        "\nWHOLE-OP blocking arms (device kernel ns)\n"
        + "\n".join(f"    {k:<28} {v:>9.0f} ns" for k, v in RESULTS.items())
    )


@pytest.mark.parametrize("ablate_read", [False, True], ids=["full", "no_read"])
@pytest.mark.parametrize("cap", [None, 4], ids=["chunk8_current", "chunk4_coalesceable"])
def test_reshard_blocking(device, cap, ablate_read):
    """`no_read` strips the READ payload and keeps every barrier / CB handshake /
    compute / write, so it is the NON-READ cost of the blocking — the part a
    coalesced gather cannot remove."""
    shape, src, dst = B.RESHARD_SHAPE
    original = pd.wt_cap
    if cap is not None:
        pd.wt_cap = lambda *a, **k: min(cap, original(*a, **k))
    try:
        ns = B._measure(
            device,
            shape=shape,
            dtype=ttnn.bfloat16,
            in_mem_config=B._width_shard(shape, src),
            out_mem_config=B._height_shard(shape, dst),
            ablate=dict(dm_read=1) if ablate_read else None,
            label=f"reshard/wt_cap={cap}/no_read={ablate_read}",
        )
    finally:
        pd.wt_cap = original
    RESULTS[f"reshard_cap{cap}_{'no_read' if ablate_read else 'full'}"] = ns
