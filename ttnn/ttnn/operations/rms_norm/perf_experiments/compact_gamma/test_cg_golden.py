# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden correctness sweep of the compact_gamma clone (graduation defaults) through the golden harness.

    CGG_FILTER=loose scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/compact_gamma/test_cg_golden.py --import-mode=prepend

Same params / marks as eval/golden_tests/rms_norm/test_golden.py (xfail / skip marks kept); every dispatch goes through
cg_descriptor.create_program_descriptor. CGG_FILTER: "loose" (all LOOSE_CASES), "gamma_wide" (cartesian gamma cells
with W >= 4096 or sharded), "all".
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.rms_norm.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("cg_descriptor", HERE / "cg_descriptor.py")
cgd = importlib.util.module_from_spec(_spec)
sys.modules["cg_descriptor"] = cgd
_spec.loader.exec_module(cgd)
op_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")
pkg = importlib.import_module("ttnn.operations.rms_norm")
import ttnn  # noqa: E402

pytestmark = pytest.mark.use_module_device
FILTER = os.environ.get("CGG_FILTER", "loose")


def _gamma_wide(p):
    inputs, axes = p.values[0], p.values[1]
    if axes.get("gamma_mode") != "gamma":
        return False
    return inputs[0][-1] >= 4096 or axes["memory_layout"] != ttnn.TensorMemoryLayout.INTERLEAVED


CART = (
    [
        p
        for p in parametrize_cases(TARGET, INPUTS, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID)
        if FILTER == "all" or (FILTER == "gamma_wide" and _gamma_wide(p))
    ]
    if FILTER in ("all", "gamma_wide")
    else []
)
LOOSE = (
    parametrize_loose_cases(LOOSE_CASES, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID)
    if FILTER in ("all", "loose")
    else []
)


@pytest.fixture(autouse=True)
def _clone(monkeypatch):
    if os.environ.get("CGG_OP") == "1":  # baseline: the unmodified op through the same params
        return
    monkeypatch.setattr(op_mod, "create_program_descriptor", cgd.create_program_descriptor)


@pytest.mark.parametrize("inputs,axes", CART)
def test_op(inputs, axes, device):
    run_rms_norm(inputs, device=device, **axes)


@pytest.mark.parametrize("inputs,axes,extras", LOOSE)
def test_op_loose(inputs, axes, extras, device):
    run_rms_norm(inputs, device=device, extras=extras, **axes)
