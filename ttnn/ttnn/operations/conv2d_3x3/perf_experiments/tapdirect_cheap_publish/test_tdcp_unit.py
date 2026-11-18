# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's unit tests, unchanged, retargeted at this experiment's op copy (op_copy/: 3ca4eb28f46 + the padded
slab, PADDED_SLAB = True).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tapdirect_cheap_publish/test_tdcp_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdcp_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
SWAPPED = {
    name: retarget(_BASE + name, globals())
    for name in (
        "test_conv2d_3x3",
        "test_conv2d_3x3_block_sharded",
        "test_conv2d_3x3_height_sharded",
        "test_conv2d_3x3_layouts",
        "test_conv2d_3x3_kw_fold",
        "test_conv2d_3x3_perf",
        "test_conv2d_3x3_precision_baseline",
    )
}

# Experiment-only: log every program that takes the padded slab (runs/unit_padded.jsonl), to count the unit shapes on
# the new path.
import json  # noqa: E402
import os  # noqa: E402

from tdcp_copy import copy_modules  # noqa: E402

_op, _pd = copy_modules()
_LOG = Path(__file__).resolve().parent / "runs" / "unit_padded.jsonl"
_real_create = _op.create_program_descriptor


def _logged_create(*args, **kwargs):
    desc = _real_create(*args, **kwargs)
    info = getattr(_pd, "LAST_PADDED_INFO", None)
    if info and info.get("on"):
        with open(_LOG, "a") as f:
            f.write(json.dumps({"test": os.environ.get("PYTEST_CURRENT_TEST", ""), **info}) + "\n")
    return desc


_op.create_program_descriptor = _logged_create
