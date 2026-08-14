# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does the strip form COMPOSE with the padded (R_PAD) reader?

The op's shipped padded-into-a-local-shard bench row
([1,1,2040,256] -> [1,1,2048,256] on 8 destination cores) reads its source from
INTERLEAVED DRAM, so `src_row_pages == 1`: there is no cross-core gather there at
all and the strip form has nothing to act on (consecutive rows of an interleaved
tensor live in different banks, so a multi-row transfer is not contiguous).

The composition question is therefore asked of the geometry where it is real: the
SAME padding against a WIDTH-sharded L1 source, i.e. a padded reshard. Raggedness
is per BLOCK, so the fast strip covers every whole tile-row and only the H tail
falls back — into the same STRIP-MAJOR addresses, so the compute side never
learns the difference.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/gather_strip/test_pad.py
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.gather_strip import strip_bench as S

# name -> (padded target, h_in, dtype, src_cores, dst_cores)
CASES = {
    # H tail only: 2040 real rows in a 2048 target -> tile-row 63 is ragged, and
    # it lives on ONE core (core 7). Every other block is a whole strip.
    "pad_h_tail": ([1, 1, 2048, 256], 2040, ttnn.bfloat16, 2, 8),
    # Deep pad — HALF the target is pad. Cores 0-3 are all-real (pure strip),
    # core 4 carries the one ragged block, cores 5-7 are WHOLE-pad (pure fill, no
    # read at all). The worst case the fallback can be asked for.
    "pad_h_deep": ([1, 1, 1024, 256], 520, ttnn.bfloat16, 2, 8),
}

RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    lines = ["", "=" * 96, "GATHER STRIP x PAD — end-to-end device kernel ns", "=" * 96]
    for case, arms in RESULTS.items():
        row = arms.get("row")
        lines.append(f"  {case}")
        for name, ns in arms.items():
            speed = f"  {row / ns:.2f}x" if row and ns else ""
            lines.append(f"      {name:<12} {ns:>10.0f} ns{speed}")
    logger.info("\n".join(lines) + "\n" + "=" * 96)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("arm", ("row", "strip", "strip_fine"))
def test_pad(device, case, arm):
    shape, h_in, dtype, src_cores, dst_cores = CASES[case]
    ns = S.run(
        device,
        shape=shape,
        dtype=dtype,
        src_cores=src_cores,
        dst_cores=dst_cores,
        arm=arm,
        h_in=h_in,
        label=f"{case}/{arm}",
    )
    RESULTS.setdefault(case, {})[arm] = ns
