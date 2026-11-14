# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The single-device golden suite (eval/golden_tests/mhc_post/test_golden.py test_op + test_op_loose) on the
experiment copy mhc_post_fsm (MHC_FSM_VARIANT=base: the unchanged copy). Domain check of the fused mix.

    scripts/run_safe_pytest.sh --run-all <this> --import-mode=prepend -o addopts=""
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from eval.golden_harness import parametrize_cases, parametrize_loose_cases  # noqa: E402
from eval.golden_tests.mhc_post import helpers as H  # noqa: E402
from eval.golden_tests.mhc_post.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET  # noqa: E402

_m = importlib.import_module("mhc_post_fsm" if os.environ.get("MHC_FSM_VARIANT", "fsm") == "fsm" else "mhc_post_base")
H.mhc_post = _m.mhc_post  # helpers' default op


def _sid(inputs):
    return "X" + "x".join(str(d) for d in inputs[1])


def _feed(p):
    return p.values[1]["feed"] == "on"


@pytest.mark.parametrize(
    "inputs,axes",
    [
        p
        for p in parametrize_cases(
            TARGET, INPUTS, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.EXCLUSIONS, INVALID, shape_id=_sid
        )
        if not _feed(p)
    ],
)
def test_op(inputs, axes, device):
    H.run_mhc_post(inputs, device=device, **axes)


@pytest.mark.parametrize(
    "inputs,axes,extras",
    [
        p
        for p in parametrize_loose_cases(
            LOOSE_CASES, _m.INPUT_TAGGERS, _m.SUPPORTED, _m.EXCLUSIONS, INVALID, shape_id=_sid
        )
        if not _feed(p)
    ],
)
def test_op_loose(inputs, axes, extras, device):
    H.run_mhc_post(inputs, device=device, extras=extras, **axes)
