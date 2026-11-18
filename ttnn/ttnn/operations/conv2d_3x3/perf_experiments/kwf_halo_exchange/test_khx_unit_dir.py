# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit dir (tests/ttnn/unit_tests/operations/conv2d_3x3/, perf harness excluded), unchanged, retargeted
at this experiment's op copy.

    KHX_MODE=2 scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_halo_exchange/test_khx_unit_dir.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khx_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
SWAPPED = {
    name: retarget(_BASE + name, globals())
    for name in (
        "test_conv2d_3x3",
        "test_conv2d_3x3_block_sharded",
        "test_conv2d_3x3_height_sharded",
        "test_conv2d_3x3_layouts",
        "test_conv2d_3x3_kw_fold",
        "test_conv2d_3x3_precision_baseline",
    )
}
