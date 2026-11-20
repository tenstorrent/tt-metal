# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3's kw-fold unit file plus the new round-0 self-read test (krs_unit_kw_fold.py), and the HEIGHT unit file,
retargeted at op_grad/ (KRS_OP=base: op_base/, which lacks _r0_partner_self_read: run only the existing tests there).

    KRS_OP=grad scripts/run_safe_pytest.sh --run-all --precompile --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_round0_sources/test_krs_unit.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from krs_copy import retarget  # noqa: E402

_BASE = "tests.ttnn.unit_tests.operations.conv2d_3x3."
_WHICH = os.environ.get("KRS_OP", "grad")
_MODULES = [
    m for m in os.environ.get("KRS_UNIT_MODULES", "krs_unit_kw_fold,test_conv2d_3x3_height_sharded").split(",") if m
]
SWAPPED = {name: retarget(name if name.startswith("krs_") else _BASE + name, globals(), _WHICH) for name in _MODULES}
