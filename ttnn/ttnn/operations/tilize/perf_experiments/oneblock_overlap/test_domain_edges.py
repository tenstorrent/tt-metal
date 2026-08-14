# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The two cells a blocking predicate must not be proposed without.

1. `d_smallest` collapsed to ONE core — the literal output of the rule
   `active = num_blocks // 2` on a shape that owns 2 blocks in total.
2. A SECOND one-block-per-core topology ([1,1,32,8192], 256 B read instead of
   512 B) and the focus shape at fp32 — so "applies to the NT_H==1 blocking" is
   not secretly "applies to [1,1,32,16384] bf16".

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_domain_edges.py
"""

import ttnn

from ._harness import bake_off


def test_smallest_collapsed_to_one_core(device):
    bake_off(
        device,
        "d_smallest",
        ["baseline", "cores2_notrid", "cores1", "cores1_notrid"],
        rounds=5,
        dtype=ttnn.bfloat16,
    )


def test_second_oneblock_topology(device):
    bake_off(
        device,
        "e_wide_short_half",
        ["baseline", "cores32", "cores32_d2_notrid", "cores16"],
        rounds=7,
        dtype=ttnn.bfloat16,
    )


def test_focus_fp32(device):
    bake_off(
        device,
        "b_wide_short",
        ["baseline", "cores32", "cores32_d2_notrid"],
        rounds=7,
        dtype=ttnn.float32,
    )
