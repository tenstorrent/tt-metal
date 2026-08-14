# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS shape — read-trid vs write-trid, at 2 blocks per core.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_focus_trid_half.py
"""

import ttnn

from ._harness import bake_off


def test_focus_trid_half(device):
    bake_off(
        device,
        "b_wide_short",
        ["baseline", "cores32", "cores32_no_readtrid", "cores32_no_writetrid", "cores32_d2_notrid"],
        rounds=9,
        dtype=ttnn.bfloat16,
    )
