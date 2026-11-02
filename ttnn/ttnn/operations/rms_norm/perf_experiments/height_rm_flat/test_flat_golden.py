# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden correctness sweep of the height_rm_flat host plan over every golden HEIGHT_SHARDED ROW_MAJOR cell.

    [HRFG_FORCE=1] scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/height_rm_flat/test_flat_golden.py --import-mode=prepend -s

Same params / marks as eval/golden_tests/rms_norm/test_golden.py, filtered to layout ROW_MAJOR x HEIGHT_SHARDED; every
dispatch goes through flat_program_descriptor.create_program_descriptor (graduation defaults). HRFG_FORCE=1 routes
every expressible + L1-fitting cell to the flat path (correctness of the flat kernels beyond the cost-model gate).
HRFG_MODULE=grad runs graduation/rms_norm_program_descriptor.py (the op's descriptor with the flat regime merged in,
kernels from graduation/kernels) instead of this dir's flat_program_descriptor.py.
Prints `HRFG_ROUTE <flat|tilized>` per dispatch.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.rms_norm.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HERE = Path(__file__).parent
GRAD = os.environ.get("HRFG_MODULE") == "grad"  # graduation/rms_norm_program_descriptor.py (the op file as proposed)
_src = HERE / "graduation" / "rms_norm_program_descriptor.py" if GRAD else HERE / "flat_program_descriptor.py"
_spec = importlib.util.spec_from_file_location("hrf_descriptor", _src)
fpd = importlib.util.module_from_spec(_spec)
sys.modules["hrf_descriptor"] = fpd
_spec.loader.exec_module(fpd)
op_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")
pkg = importlib.import_module("ttnn.operations.rms_norm")

pytestmark = pytest.mark.use_module_device


def _rm_height(p):
    axes = p.values[1]
    return (
        axes.get("layout") == ttnn.ROW_MAJOR_LAYOUT
        and axes.get("memory_layout") == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    )


CART = [
    p
    for p in parametrize_cases(TARGET, INPUTS, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID)
    if _rm_height(p)
]
LOOSE = [
    p
    for p in parametrize_loose_cases(LOOSE_CASES, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID)
    if _rm_height(p)
]


@pytest.fixture(autouse=True)
def _clone(monkeypatch):
    monkeypatch.setattr(fpd, "RM_FLAT_FORCE" if GRAD else "FLAT_FORCE", os.environ.get("HRFG_FORCE") == "1")
    real_flat = fpd.create_flat_program_descriptor

    def flat(*a, **k):
        print("HRFG_ROUTE flat")
        return real_flat(*a, **k)

    owner, name = (fpd, "_create_tilized_program_descriptor") if GRAD else (fpd.pd, "create_program_descriptor")
    real_base = getattr(owner, name)

    def base(*a, **k):
        print("HRFG_ROUTE tilized")
        return real_base(*a, **k)

    monkeypatch.setattr(fpd, "create_flat_program_descriptor", flat)
    monkeypatch.setattr(owner, name, base)
    monkeypatch.setattr(op_mod, "create_program_descriptor", fpd.create_program_descriptor)


@pytest.mark.parametrize("inputs,axes", CART)
def test_op(inputs, axes, device):
    run_rms_norm(inputs, device=device, **axes)


@pytest.mark.parametrize("inputs,axes,extras", LOOSE)
def test_op_loose(inputs, axes, extras, device):
    run_rms_norm(inputs, device=device, extras=extras, **axes)
