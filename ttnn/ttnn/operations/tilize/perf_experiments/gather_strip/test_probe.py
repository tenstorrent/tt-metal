# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""CONTENTION PROBE — how much of the strip win is real headroom?

Perf 1's load-bearing mechanism correction: `reader_issue >> reader_barrier` does
NOT prove "RISC-bound on transaction count", because `noc_async_read`
back-pressures INSIDE the issue loop when the fabric is saturated. The same
per-core issue loop cost 10,711 ns with 1 destination core and 17,633 ns with 8 —
the extra 6.9 us was contention on the source cores' shared L1 egress.

So: hold the PER-CORE work exactly constant (4 blocks of identical geometry, same
2-core source) and vary only how many destination cores are pulling at once.

  * If the strip win survives at dst=1, it is real per-core headroom (fewer
    RISC-issued transactions and larger transfers), not rescheduling.
  * The extra win that only appears at dst=8 is contention relief — real, but a
    different mechanism, and it evaporates on a thinner grid.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/gather_strip/test_probe.py
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.gather_strip import strip_bench as S

# dst cores -> shape that gives EVERY core exactly 4 blocks (128 rows) of the
# focus geometry, against the same 2-core WIDTH-sharded source.
PROBE = {f"dst{n}": ([1, 1, 128 * n, 256], ttnn.bfloat16, 2, n) for n in (1, 2, 4, 8)}

RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    lines = ["", "=" * 96, "CONTENTION PROBE — identical per-core work, varying destination cores", "=" * 96]
    for case, arms in RESULTS.items():
        row = arms.get("row")
        lines.append(f"  {case}")
        for name, ns in arms.items():
            speed = f"  {row / ns:.2f}x" if row and ns else ""
            lines.append(f"      {name:<12} {ns:>10.0f} ns{speed}")
    logger.info("\n".join(lines) + "\n" + "=" * 96)


@pytest.mark.parametrize("case", list(PROBE))
@pytest.mark.parametrize("arm", ("row", "strip", "strip_fine"))
def test_probe(device, case, arm):
    shape, dtype, src_cores, dst_cores = PROBE[case]
    ns = S.run(
        device,
        shape=shape,
        dtype=dtype,
        src_cores=src_cores,
        dst_cores=dst_cores,
        arm=arm,
        label=f"{case}/{arm}",
    )
    RESULTS.setdefault(case, {})[arm] = ns
