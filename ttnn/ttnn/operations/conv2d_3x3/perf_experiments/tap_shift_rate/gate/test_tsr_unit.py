# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""conv2d_3x3's BLOCK unit tests, unchanged, retargeted at grad_copy/ (HEAD + graduation.patch).

    scripts/run_safe_pytest.sh [--dev] --run-all --precompile --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tap_shift_rate/gate/test_tsr_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tsr_copies import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded", globals())
assert SWAPPED, "nothing retargeted"
