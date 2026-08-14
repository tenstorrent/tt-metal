# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS shape — is the depth-4 edge the CB DEPTH or the CORE COUNT?

`baseline_d4` is the discriminator: at one block per core a deeper CB has nothing
to hold, so if it matches `baseline` the edge belongs to the 2-blocks-per-core
configuration; if it moves, the depth was never about overlap at all.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_focus_depth.py
"""

import ttnn

from ._harness import bake_off


def test_focus_depth(device):
    bake_off(
        device,
        "b_wide_short",
        ["baseline", "baseline_d4", "cores32_d2_notrid", "cores32_d4", "cores16_d4"],
        rounds=9,
        dtype=ttnn.bfloat16,
    )
