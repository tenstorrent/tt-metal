# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""FINAL: the RECOMMENDED option (`cores32_no_readtrid`) vs the shipped op,
re-measured on every regime in the sweep.

Recommended option = cap the W_BLOCKS split to `num_blocks_total / 2` cores so
each owns exactly TWO blocks (WT_CHUNK, n_chunks and the read transfer all
UNCHANGED) and compile the reader with `read_trid = 0`. The write-side trid stays
ON — the half-and-half arms showed the read half is the whole cost.

Every regime here gets the SAME two arms so the domain is a like-for-like table,
not a mix of arm sets.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_final_domain.py
"""

import pytest
import ttnn

from ._harness import bake_off

_ARMS = ["baseline", "cores32_no_readtrid"]


def test_final_focus(device):
    bake_off(device, "b_wide_short", _ARMS, rounds=15, dtype=ttnn.bfloat16)


@pytest.mark.parametrize("shape_key", ["e_wide_short_half", "a_square", "c_multiblock"])
def test_final_sweep(device, shape_key):
    bake_off(device, shape_key, _ARMS, rounds=7, dtype=ttnn.bfloat16)


def test_final_focus_fp32(device):
    bake_off(device, "b_wide_short", _ARMS, rounds=7, dtype=ttnn.float32)
