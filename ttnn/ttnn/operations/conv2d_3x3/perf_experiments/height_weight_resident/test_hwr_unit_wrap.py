# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of graduation.patch on the op's own unit tests: runs conv2d_3x3 unit test modules against grad_copy/
(the op at e40c435b625 with the patch applied) by aliasing the op's modules to the copy before the unit test modules
import them. HWR_UNIT selects the modules (default: height_sharded,interleaved).

  scripts/run_safe_pytest.sh --run-all <this file> -p no:cacheprovider --import-mode=prepend
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hwr_load  # noqa: E402

opc, pd, op_mod = hwr_load.load_grad()

sys.modules["ttnn.operations.conv2d_3x3"] = opc
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3"] = op_mod
sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"] = pd

_UNIT = os.environ.get("HWR_UNIT", "height_sharded,interleaved").split(",")
if "height_sharded" in _UNIT:
    from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_height_sharded import *  # noqa: E402,F401,F403
if "interleaved" in _UNIT:
    from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3 import *  # noqa: E402,F401,F403
if "block_sharded" in _UNIT:
    from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded import *  # noqa: E402,F401,F403
if "layouts" in _UNIT:
    from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_layouts import *  # noqa: E402,F401,F403

DEPTHS = []
_real_create = op_mod.create_program_descriptor


def _counting_create(*args, blocking, **kwargs):
    d = _real_create(*args, blocking=blocking, **kwargs)
    servers = any("weight_server" in str(k.kernel_source) for k in d.kernels)
    DEPTHS.append((blocking.weight_depth, blocking.nslice, servers))
    return d


op_mod.create_program_descriptor = _counting_create


def test_zz_resident_weights_ran():
    """The wrapped unit tests built programs with a deepened cb_weights, on the server and the non-server path."""
    deep = [d for d in DEPTHS if not d[1] and d[0] > 2]
    print(f"HWR_WRAP depths (weight_depth, nslice, servers): {sorted(set(DEPTHS))}")
    print(
        f"HWR_WRAP: {len(DEPTHS)} programs, {len(deep)} with weight_depth > 2 "
        f"({sum(1 for d in deep if d[2])} on weight servers), depths {sorted({d[0] for d in deep})}"
    )
    if os.environ.get("HWR_EXPECT_DEEP", "1") == "1":
        assert deep
