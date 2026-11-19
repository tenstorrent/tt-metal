# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The op's native-shard unit tests (test_all_gather_rms_norm_native_shard.py) against agrms_copy with the recommended
notify variant on by default (override any knob by env). The test spies on the descriptor module's `plan`, so that
module name is pointed at the copy's descriptor before the test module imports it. Run with --import-mode=prepend:

    scripts/run_safe_pytest.sh --no-precompile --run-all <this file> --import-mode=prepend
"""

import os
import sys
from pathlib import Path

for _k, _v in {
    "AGRMS_NOTIFY_SPLIT_FINAL": "1",
    "AGRMS_NOTIFY_HOIST": "1",
    "AGRMS_NOTIFY_PLACE": "1",
}.items():
    os.environ.setdefault(_k, _v)

sys.path.insert(0, str(Path(__file__).parent))
import agrms_copy  # noqa: E402
import agrms_copy.all_gather_rms_norm_program_descriptor as _copy_pd  # noqa: E402
import ttnn.operations.all_gather_rms_norm as _real  # noqa: E402

_real.all_gather_rms_norm = agrms_copy.all_gather_rms_norm  # the test modules bind the name at import
sys.modules["ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor"] = _copy_pd
_real.all_gather_rms_norm_program_descriptor = _copy_pd

from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm_native_shard import *  # noqa: E402,F401,F403
from tests.ttnn.unit_tests.operations.all_gather_rms_norm import (
    test_all_gather_rms_norm_native_shard as _t,
)  # noqa: E402

assert _t.PD is _copy_pd, "native tests would spy on the real op's descriptor"
assert _t.run_case.__globals__["all_gather_rms_norm"] is agrms_copy.all_gather_rms_norm, "would run the real op"
