# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The op's unit tests (tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm.py) run against
agrms_copy with the recommended port_stream_rate variant (RELAY_FIRST + STATEFUL + CACHED_SLOTS) on by default (override any knob by env). Run with
--import-mode=prepend (the repo's importlib mode resolves this dir as a ttnn.ttnn subpackage):

    scripts/run_safe_pytest.sh --no-precompile --run-all <this file> --import-mode=prepend
"""

import os
import sys
from pathlib import Path

for _k, _v in {
    "AGRMS_PSR_DEFINES": "AGRMS_PSR_RELAY_FIRST,AGRMS_PSR_STATEFUL,AGRMS_PSR_CACHED_SLOTS",
}.items():
    os.environ.setdefault(_k, _v)

sys.path.insert(0, str(Path(__file__).parent))
import agrms_copy  # noqa: E402
import ttnn.operations.all_gather_rms_norm as _real  # noqa: E402

_real.all_gather_rms_norm = agrms_copy.all_gather_rms_norm  # the test module binds the name at import

from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm import *  # noqa: E402,F401,F403
from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm import (  # noqa: E402
    all_gather_rms_norm as _bound,
)

assert _bound is agrms_copy.all_gather_rms_norm, "unit tests would run the real op"
