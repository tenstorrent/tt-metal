# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""DOMAIN sweep, 2x2 — is the multi-block-regime gain the CORE CAP or just B8-off?

On the focus shape the two were separated cleanly (`baseline_notrid` landed flat
on `baseline`, so the whole gain belongs to the core cap). Shapes (a) and (c)
already own ~4 blocks per core, so the same 2x2 has to be run there before any
"applies everywhere" claim: without `baseline_notrid` the sweep cannot tell a
core-cap win from a plain "turn the trid double-issue off" win.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/oneblock_overlap/test_sweep_isolate.py
"""

import pytest
import ttnn

from ._harness import bake_off

_ARMS_2X2 = ["baseline", "baseline_notrid", "cores32", "cores32_d2_notrid"]


@pytest.mark.parametrize("shape_key", ["a_square", "c_multiblock"])
def test_sweep_isolate(device, shape_key):
    bake_off(device, shape_key, _ARMS_2X2, rounds=5, dtype=ttnn.bfloat16)
