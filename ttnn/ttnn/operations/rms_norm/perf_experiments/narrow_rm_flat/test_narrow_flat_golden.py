# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden correctness sweep of the graduation candidate (graduate/rms_norm_narrow_flat_program_descriptor.py) over every
golden ROW_MAJOR x WIDTH / BLOCK_SHARDED cell (cartesian + loose), same params / marks as
eval/golden_tests/rms_norm/test_golden.py.

    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/rms_norm/perf_experiments/narrow_rm_flat/test_narrow_flat_golden.py --import-mode=prepend -s

Every dispatch goes through the graduation hook: the narrow-flat program when it applies, else the op's own descriptor.
Prints `NFG_ROUTE <narrow_flat|op>` per dispatch.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.rms_norm.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from eval.golden_tests.rms_norm.helpers import run_rms_norm

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "nfg_grad", HERE / "graduate" / "rms_norm_narrow_flat_program_descriptor.py"
)
gpd = importlib.util.module_from_spec(_spec)
sys.modules["nfg_grad"] = gpd
_spec.loader.exec_module(gpd)
op_mod = importlib.import_module("ttnn.operations.rms_norm.rms_norm")
pkg = importlib.import_module("ttnn.operations.rms_norm")
BASE = op_mod.create_program_descriptor

pytestmark = pytest.mark.use_module_device


def _rm_wb(p):
    axes = p.values[1]
    return axes.get("layout") == ttnn.ROW_MAJOR_LAYOUT and axes.get("memory_layout") in (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )


CART = [
    p for p in parametrize_cases(TARGET, INPUTS, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID) if _rm_wb(p)
]
LOOSE = [
    p
    for p in parametrize_loose_cases(LOOSE_CASES, pkg.INPUT_TAGGERS, pkg.SUPPORTED, pkg.EXCLUSIONS, INVALID)
    if _rm_wb(p)
]


def _hooked(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    d = gpd.create_narrow_flat_program_descriptor(
        input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=compute_kernel_config
    )
    if d is not None:
        print("NFG_ROUTE narrow_flat")
        return d
    print("NFG_ROUTE op")
    return BASE(input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=compute_kernel_config)


@pytest.fixture(autouse=True)
def _clone(monkeypatch):
    monkeypatch.setattr(op_mod, "create_program_descriptor", _hooked)


@pytest.mark.parametrize("inputs,axes", CART)
def test_op(inputs, axes, device):
    run_rms_norm(inputs, device=device, **axes)


@pytest.mark.parametrize("inputs,axes,extras", LOOSE)
def test_op_loose(inputs, axes, extras, device):
    run_rms_norm(inputs, device=device, extras=extras, **axes)
