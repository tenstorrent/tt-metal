# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS shape, TIGHT A/B on the four contenders only.

The 11-arm menu put every no-regression arm inside this shape's documented
+-4-6% spread, so the win/null call sits on the noise band and needs more
samples per arm on fewer arms. 9 interleaved rounds.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_focus_tight.py
"""

import ttnn

from ._harness import bake_off


def test_focus_tight(device):
    bake_off(
        device,
        "b_wide_short",
        ["baseline", "cores32", "cores16", "cores32_d4"],
        rounds=9,
        dtype=ttnn.bfloat16,
    )
