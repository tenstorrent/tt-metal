# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's kw-fold unit tests, unchanged, retargeted at an op copy (KFP_OP, default op_copy).

    KFP_OP=op_copy3 scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_final_pass/test_kfp_unit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfp_copy import retarget  # noqa: E402

SWAPPED = retarget("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_kw_fold", globals())


def test_kfp_retargeted():
    import os

    from kfp_copy import copy_modules

    op_mod, pd_mod = copy_modules()
    assert {"conv2d_3x3", "prepare_conv2d_weights"} <= set(SWAPPED), SWAPPED
    assert os.environ.get("KFP_OP", "op_copy") in str(pd_mod.KERNEL_DIR), pd_mod.KERNEL_DIR
    print("KFP_RETARGET", SWAPPED, pd_mod.KERNEL_DIR)
