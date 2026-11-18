# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's HEIGHT-sharded guard set (test_conv2d_3x3_perf.py, test_conv2d_3x3_height_sharded_perf: the 14
Panoptic HEIGHT layers at the default HiFi4 config), unchanged, retargeted at one op copy and filtered by env.

    KWCR_COPY=op_cand KWCR_HCASES=res2_conv2,center_head1 scripts/run_safe_pytest.sh --profile --run-all \
        --import-mode=prepend ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds/test_kwcr_height.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwcr_copy import retarget  # noqa: E402

_NS = {}
SWAPPED = retarget(
    "tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf", _NS, os.environ.get("KWCR_COPY", "op_cand")
)
from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf import HEIGHT_SHAPES  # noqa: E402

_want = [s for s in os.environ.get("KWCR_HCASES", "").split(",") if s]
_SHAPES = [s for s in HEIGHT_SHAPES if not _want or s[0] in _want]


@pytest.mark.parametrize("layer,case,dtype,layout,shard_shape,num_cores", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_kwcr_height(layer, case, dtype, layout, shard_shape, num_cores, device):
    _NS["test_conv2d_3x3_height_sharded_perf"](layer, case, dtype, layout, shard_shape, num_cores, device)
