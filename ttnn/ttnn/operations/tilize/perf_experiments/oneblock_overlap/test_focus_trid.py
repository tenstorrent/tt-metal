# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS shape — why 2 blocks/core only pays with the B8 trid double-issue OFF.

The trid path (tilize_reader.cpp, `read_trid`) issues block i+1's reads BEFORE
block i's barrier, so block i is PUSHED only after block i+1 has been issued. At
4+ blocks per core that cost is amortized into a steady state; at exactly TWO it
is pure added latency on the ONLY handoff there is — compute (and therefore the
writer) starts one whole issue loop later. This arm set prices that directly.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_focus_trid.py
"""

import ttnn

from ._harness import bake_off


def test_focus_trid(device):
    bake_off(
        device,
        "b_wide_short",
        ["baseline", "cores32", "cores32_d2_notrid", "cores16", "cores16_notrid"],
        rounds=9,
        dtype=ttnn.bfloat16,
    )
