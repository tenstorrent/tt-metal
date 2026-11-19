# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's BLOCK_SHARDED unit tests (tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py),
unchanged, retargeted at this experiment's op copy (variant from BWR_* environment variables)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bwr_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded", globals())
assert "_pd" in SWAPPED and "_op_module" in SWAPPED, SWAPPED
