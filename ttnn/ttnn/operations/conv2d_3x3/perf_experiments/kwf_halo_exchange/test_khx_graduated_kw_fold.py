# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tests_copy/test_conv2d_3x3_kw_fold.py.txt (the unit file with the halo-exchange test that tests.patch adds; kept as
.txt so no file under ttnn/ imports torch globally), materialized into a temp dir and retargeted at this experiment's
op copy.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_halo_exchange/test_khx_graduated_kw_fold.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="khx_tests_"))
shutil.copy(HERE / "tests_copy" / "test_conv2d_3x3_kw_fold.py.txt", _TMP / "khx_graduated_kw_fold_tests.py")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_TMP))

from khx_copy import retarget  # noqa: E402

SWAPPED = retarget("khx_graduated_kw_fold_tests", globals())
