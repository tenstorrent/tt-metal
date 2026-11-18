# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the recommended variant on the op's own unit tests: runs the conv2d_3x3 unit test modules against
this experiment's op_copy with the weight-server knobs on (HWM_* below), by aliasing the op's modules to the copy
before the unit test modules import them.

  scripts/run_safe_pytest.sh --run-all <this file> -p no:cacheprovider --import-mode=prepend
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hwm_load  # noqa: E402

opc, pd, op_mod = hwm_load.load()

# The recommended variant: one weight server per channel rectangle on a spare core, 6 chunks, protocol v2 (op_copy
# knobs; the graduation tree, HWM_COPY=grad, has it on by default).
if not hwm_load.GRAD:
    pd.HWM_SENDERS = 1
    pd.HWM_PER_RECT = True
    pd.HWM_SERVER = True
    pd.HWM_V2 = True
    pd.HWM_CHUNKS = 6

sys.modules["ttnn.operations.conv2d_3x3"] = opc
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3"] = op_mod
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"] = pd

from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3 import *  # noqa: E402,F401,F403
from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_height_sharded import *  # noqa: E402,F401,F403
from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_layouts import *  # noqa: E402,F401,F403

SERVER_PROGRAMS = []
_real_create = op_mod.create_program_descriptor


def _counting_create(*args, **kwargs):
    d = _real_create(*args, **kwargs)
    SERVER_PROGRAMS.append(any("server" in str(k.kernel_source) for k in d.kernels))
    return d


op_mod.create_program_descriptor = _counting_create


def test_zz_weight_servers_ran():
    """The wrapped unit tests built programs on the weight-server path (not only the baseline fallback)."""
    print(f"HWM_WRAP: {sum(SERVER_PROGRAMS)} of {len(SERVER_PROGRAMS)} programs used weight servers")
    assert sum(SERVER_PROGRAMS) > 0
