# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS shape — the DECIDING A/B, 15 interleaved rounds on four arms.

The three earlier sessions agree on the ordering but the ratios move with the
baseline's own +-6% drift, so the headline claim is settled here with more
samples on fewer arms.

  baseline            the op as it ships (64 cores, 1 block/core, B8 trid ON)
  baseline_notrid     control: at ONE block per core the trid double-issue has
                      no second block to overlap, so this MUST land on baseline
  cores32_d2_notrid   the candidate: same WT_CHUNK / same 512 B read, 32 cores,
                      2 blocks/core, trid OFF
  cores32_d4          the candidate + a 4-deep CB (depth != 2 forces trid off)

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_focus_final.py
"""

import ttnn

from ._harness import bake_off


def test_focus_final(device):
    bake_off(
        device,
        "b_wide_short",
        ["baseline", "baseline_notrid", "cores32_d2_notrid", "cores32_d4"],
        rounds=15,
        dtype=ttnn.bfloat16,
    )
