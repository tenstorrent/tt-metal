# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FOCUS shape [1,1,32,16384] bf16 DRAM->DRAM — the full arm menu, interleaved.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_focus.py

One test = one session, every arm round-robined 5x so the shape's +-4-6% drift
hits all arms equally. Correctness (bit-exact) is gated on round 0 of every arm;
perf is measured and logged, never asserted.
"""

import ttnn

from ._harness import ARMS, bake_off


def test_focus_arm_menu(device):
    bake_off(device, "b_wide_short", list(ARMS), rounds=5, dtype=ttnn.bfloat16)
