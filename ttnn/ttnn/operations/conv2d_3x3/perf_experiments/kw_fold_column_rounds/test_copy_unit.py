# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests, unchanged, retargeted at the KWCR_COPY op copy (default op_cand; see kwcr_copy.py).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds/test_cand_unit.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwcr_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
SWAPPED = {
    name: retarget(_BASE + name, globals(), os.environ.get("KWCR_COPY", "op_cand"))
    for name in (
        "test_conv2d_3x3",
        "test_conv2d_3x3_block_sharded",
        "test_conv2d_3x3_height_sharded",
        "test_conv2d_3x3_layouts",
        "test_conv2d_3x3_perf",
        "test_conv2d_3x3_precision_baseline",
    )
}
