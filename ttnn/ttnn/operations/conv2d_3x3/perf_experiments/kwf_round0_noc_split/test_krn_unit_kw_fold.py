# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_kw_fold.py, unchanged, retargeted at this experiment's op copy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from krn_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_kw_fold", globals())
