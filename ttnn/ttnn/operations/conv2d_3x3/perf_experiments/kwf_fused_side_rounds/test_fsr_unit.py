# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's kw-fold and HEIGHT unit tests, unchanged, retargeted at op_copy/ (FSR_OP=base: op_base/).

    scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_fused_side_rounds/test_fsr_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsr_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
SWAPPED = {name: retarget(_BASE + name, globals()) for name in ("test_conv2d_3x3_kw_fold",)}
