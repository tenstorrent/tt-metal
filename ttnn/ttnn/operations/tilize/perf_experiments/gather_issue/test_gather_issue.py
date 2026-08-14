# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off: cheaper / fewer read transactions in tilize's cross-core L1 gather.

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/tilize/perf_experiments/gather_issue/test_gather_issue.py

Correctness is the ONLY pass/fail (the gather is a pure permutation, so the bar
is `torch.equal`). Perf is measured and reported, never asserted.
"""

import pytest
import ttnn
from loguru import logger

from ttnn.operations.tilize.perf_experiments.gather_issue import gather_bench as G

# name -> (shape, dtype, src_cores, dst_cores, wt_chunk)
CASES = {
    # ---- the focus plan, at the block width the production host actually picks
    "focus_c8": ([1, 1, 1024, 256], ttnn.bfloat16, 2, 8, 8),
    # ---- same plan, block width == source shard width (coalesce expressible)
    "focus_c4": ([1, 1, 1024, 256], ttnn.bfloat16, 2, 8, 4),
    # ---- the "gated reshard": a 4-core (128 B page) source
    "gated_c4": ([1, 1, 1024, 256], ttnn.bfloat16, 4, 8, 4),
    "gated_c2": ([1, 1, 1024, 256], ttnn.bfloat16, 4, 8, 2),
    # ---- domain: twice as tall (twice the blocks per core)
    "tall_c8": ([1, 1, 2048, 256], ttnn.bfloat16, 2, 8, 8),
    "tall_c4": ([1, 1, 2048, 256], ttnn.bfloat16, 2, 8, 4),
    # ---- domain: fp32 (doubles every byte quantity, halves the transaction/byte)
    "fp32_c8": ([1, 1, 1024, 256], ttnn.float32, 2, 8, 8),
    "fp32_c4": ([1, 1, 1024, 256], ttnn.float32, 2, 8, 4),
    # ---- domain: small — one block per core, where per-core overhead dominates
    "small_c8": ([1, 1, 256, 256], ttnn.bfloat16, 2, 8, 8),
    "small_c4": ([1, 1, 256, 256], ttnn.bfloat16, 2, 8, 4),
    # ---- domain: wide W (16 tile-columns), 4 slices per row at chunk 8
    "wide_c8": ([1, 1, 1024, 512], ttnn.bfloat16, 2, 8, 8),
    "wide_c16": ([1, 1, 1024, 512], ttnn.bfloat16, 2, 8, 16),
    # ---- MECHANISM PROBE, not a domain point: identical PER-CORE work (4 blocks
    # of the same geometry) with 1 / 2 destination cores instead of 8. If the
    # gather were RISC-issue-bound these must all cost the same; if the two source
    # cores' L1 egress is the bound, they get cheaper as the readers thin out.
    "probe_1core_c8": ([1, 1, 128, 256], ttnn.bfloat16, 2, 1, 8),
    "probe_2core_c8": ([1, 1, 256, 256], ttnn.bfloat16, 2, 2, 8),
    "probe_1core_c4": ([1, 1, 128, 256], ttnn.bfloat16, 2, 1, 4),
}

RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    lines = ["", "=" * 96, "GATHER ISSUE BAKE-OFF — device kernel ns (one fresh run per arm)", "=" * 96]
    for case, arms in RESULTS.items():
        base = arms.get("baseline")
        lines.append(f"  {case}")
        for name, ns in arms.items():
            speed = f"  {base / ns:.2f}x" if base and ns else ""
            lines.append(f"      {name:<10} {ns:>10.0f} ns{speed}")
    logger.info("\n".join(lines) + "\n" + "=" * 96)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("variant", list(G.VARIANTS), ids=lambda v: G.VARIANTS[v])
def test_gather(device, case, variant):
    shape, dtype, src_cores, dst_cores, wt_chunk = CASES[case]
    elem = {ttnn.bfloat16: 2, ttnn.float32: 4}[dtype]
    p = G.plan(shape, src_cores, dst_cores, wt_chunk, elem)
    if not G.variant_applicable(variant, p):
        pytest.skip(f"{G.VARIANTS[variant]} inexpressible: row_bytes={p['row_bytes']} page={p['page_bytes']}")
    label = f"{case}/{G.VARIANTS[variant]}"
    ns = G.run(
        device,
        shape=shape,
        dtype=dtype,
        src_cores=src_cores,
        dst_cores=dst_cores,
        wt_chunk=wt_chunk,
        variant=variant,
        label=label,
    )
    RESULTS.setdefault(case, {})[G.VARIANTS[variant]] = ns
