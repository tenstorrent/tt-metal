# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Domain sweep: the perf harness's 14 HEIGHT-sharded Panoptic layers (test_conv2d_3x3_perf.py::
test_conv2d_3x3_height_sharded_perf, its default compute config, bf8_b weights, bias + ReLU), each run once per op copy
in KFP_SEQ (comma list), shape by shape, in one process. PCC asserted by the harness."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfp_copy import copy_modules  # noqa: E402

import tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf as _m  # noqa: E402

_SEQ = os.environ.get("KFP_SEQ", "op_base_unz,op_copy3_unz").split(",")
_PARAMS = [pytest.param(s, v, id=f"{s[0]}-{k}-{v}") for s in _m.HEIGHT_SHAPES for k, v in enumerate(_SEQ)]


@pytest.mark.parametrize("shape,variant", _PARAMS)
def test_kfp_height_perf(shape, variant, device):
    import ttnn

    op_mod, _ = copy_modules(variant)
    for name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
        setattr(_m, name, getattr(op_mod, name))
    _m.test_conv2d_3x3_height_sharded_perf(*shape, device)
    ttnn.ReadDeviceProfiler(device)
