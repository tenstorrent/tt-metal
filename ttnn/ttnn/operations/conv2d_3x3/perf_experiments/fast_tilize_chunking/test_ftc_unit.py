# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests (every shape compiles and passes), unchanged, retargeted at this experiment's op copy.

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fast_tilize_chunking/test_ftc_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ftc_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
SWAPPED = {
    name: retarget(_BASE + name, globals())
    for name in (
        "test_conv2d_3x3",
        "test_conv2d_3x3_block_sharded",
        "test_conv2d_3x3_height_sharded",
        "test_conv2d_3x3_layouts",
    )
}
