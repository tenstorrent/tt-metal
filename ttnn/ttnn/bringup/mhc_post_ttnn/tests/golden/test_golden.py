# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Auto-parameterized golden tests for mhc_post (registry model).

test_op iterates TARGET x INPUTS; cells outside the op's SUPPORTED are
xfail-strict, INVALID cells are skipped. test_op_loose runs the
DeepSeek-V4 perf sweep in LOOSE_CASES (fixed perf-target config; extras
carry the roofline goal and, where measured, the composite baseline).

Numerical-stability regression tests live in test_regression.py.
"""

from __future__ import annotations

import pytest

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from ttnn.bringup.mhc_post_ttnn.tests.golden.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from ttnn.bringup.mhc_post_ttnn.tests.golden.helpers import run_mhc_post
from ttnn.bringup.mhc_post_ttnn import (  # type: ignore
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
)


def _shape_id(inputs):
    """`(F, X, post, comb)` shapes -> 'X…' (the others are implied by X)."""
    return "X" + "x".join(str(d) for d in inputs[1])


@pytest.mark.parametrize(
    "inputs,axes",
    parametrize_cases(
        TARGET,
        INPUTS,
        INPUT_TAGGERS,
        SUPPORTED,
        EXCLUSIONS,
        INVALID,
        shape_id=_shape_id,
    ),
)
def test_op(inputs, axes, device):
    run_mhc_post(inputs, device=device, **axes)


@pytest.mark.parametrize(
    "inputs,axes,extras",
    parametrize_loose_cases(
        LOOSE_CASES,
        INPUT_TAGGERS,
        SUPPORTED,
        EXCLUSIONS,
        INVALID,
        shape_id=_shape_id,
    ),
)
def test_op_loose(inputs, axes, extras, device):
    run_mhc_post(inputs, device=device, extras=extras, **axes)
