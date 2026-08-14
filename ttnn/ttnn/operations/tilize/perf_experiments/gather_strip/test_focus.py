# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focus bake-off — the op's reshard plan, end-to-end, all four arms.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/gather_strip/test_focus.py

Correctness (bit-exact vs torch) is the ONLY pass/fail; perf is measured.
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.gather_strip import strip_bench as S

# name -> (shape, dtype, src_cores, dst_cores, wt_chunk override)
CASES = {
    # the perf-flagged reshard plan: page = 256 B (2 tiles... 4 tiles), 2 slices
    "focus": ([1, 1, 1024, 256], ttnn.bfloat16, 2, 8, None),
    # the gated plan: a 4-core (128 B page) source -> 4 slices at wt_chunk 8
    "gated": ([1, 1, 1024, 256], ttnn.bfloat16, 4, 8, None),
}

RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    lines = ["", "=" * 96, "GATHER STRIP — END-TO-END device kernel ns (one fresh run per arm)", "=" * 96]
    for case, arms in RESULTS.items():
        base = arms.get("op") or arms.get("row")
        lines.append(f"  {case}")
        for name, ns in arms.items():
            speed = f"  {base / ns:.2f}x" if base and ns else ""
            lines.append(f"      {name:<12} {ns:>10.0f} ns{speed}")
    logger.info("\n".join(lines) + "\n" + "=" * 96)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("arm", S.ARMS)
def test_focus(device, case, arm):
    shape, dtype, src_cores, dst_cores, wt_chunk = CASES[case]
    p = S.plan(shape, dtype, src_cores, dst_cores, wt_chunk)
    if arm != "row" and arm != "op" and not p["strip_ok"]:
        pytest.skip(f"strip inexpressible: row_bytes={p['row_bytes']} page_bytes={p['page_bytes']}")
    ns = S.run(
        device,
        shape=shape,
        dtype=dtype,
        src_cores=src_cores,
        dst_cores=dst_cores,
        arm=arm,
        wt_chunk=wt_chunk,
        label=f"{case}/{arm}",
    )
    RESULTS.setdefault(case, {})[arm] = ns
