# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""test_conv2d_3x3_kw_fold.py with graduation_tests.patch applied (kwf_unit_patched.py), retargeted at op_copy/
(FSR_OP=base: op_base/)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsr_copy import retarget  # noqa: E402

SWAPPED = retarget("kwf_unit_patched", globals())
