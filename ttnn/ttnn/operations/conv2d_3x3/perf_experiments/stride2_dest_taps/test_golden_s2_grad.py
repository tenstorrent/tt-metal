# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The golden suite's stride-2 cells (eval/golden_tests/conv2d_3x3 test_op + test_op_loose, stride == 2: every cell the
S2_TAPS change can reach) on grad/ (HEAD + graduation.patch; S2_GOLDEN_PKG=op_base for HEAD): the golden helpers' op
entry points are pointed at the copy for each test."""

import importlib
import os

import pytest

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.conv2d_3x3 import axes as golden_axes
from eval.golden_tests.conv2d_3x3 import helpers
from eval.golden_tests.conv2d_3x3.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from eval.golden_tests.conv2d_3x3.helpers import run_conv2d_3x3
from ttnn.operations.conv2d_3x3 import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED  # type: ignore

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.stride2_dest_taps." + os.environ.get("S2_GOLDEN_PKG", "grad")
G = importlib.import_module(_PKG + ".conv2d_3x3")


@pytest.fixture(autouse=True)
def use_copy(monkeypatch):
    monkeypatch.setattr(helpers, "_conv2d_3x3_raw", G.conv2d_3x3)
    monkeypatch.setattr(helpers, "prepare_conv2d_bias", G.prepare_conv2d_bias)
    monkeypatch.setattr(helpers, "prepare_conv2d_weights", G.prepare_conv2d_weights)
    monkeypatch.setattr(golden_axes, "_raw", G.conv2d_3x3)


def _shape_id(inputs):
    (H, W, c_in), c_out, padding, stride, dilation = inputs
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{padding}_s{stride}_d{dilation}"


def _stride2(params):
    return [p for p in params if p.values[0][3] == 2]


@pytest.mark.parametrize(
    "inputs,axes",
    _stride2(parametrize_cases(TARGET, INPUTS, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID, shape_id=_shape_id)),
)
def test_op(inputs, axes, device):
    run_conv2d_3x3(inputs, device=device, **axes)


@pytest.mark.parametrize(
    "inputs,axes,extras",
    _stride2(parametrize_loose_cases(LOOSE_CASES, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID, shape_id=_shape_id)),
)
def test_op_loose(inputs, axes, extras, device):
    run_conv2d_3x3(inputs, device=device, extras=extras, **axes)
