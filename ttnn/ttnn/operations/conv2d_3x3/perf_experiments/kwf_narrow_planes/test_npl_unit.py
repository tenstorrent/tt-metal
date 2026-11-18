# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests (the whole unit dir except the perf / precision-baseline harnesses), unchanged,
retargeted at op_copy/.

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_narrow_planes/test_npl_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npl_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
SWAPPED = {
    name: retarget(_BASE + name, globals())
    for name in (
        "test_conv2d_3x3",
        "test_conv2d_3x3_block_sharded",
        "test_conv2d_3x3_height_sharded",
        "test_conv2d_3x3_layouts",
        "test_conv2d_3x3_kw_fold",
    )
}
