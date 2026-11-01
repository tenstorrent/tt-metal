# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""dram_stream_bw bake-off: the op copy in ./dsb_op driven through the real entry point.

LN_KNOBS="NAME=int,..." overrides ./dsb_op/descriptor.py constants (same pattern as test_perf_bench.py).
LN_COPY=0 runs the real op instead of the copy (sanity: copy at default knobs == real op).

  test_perf_case / test_guard_case : one call per perf / guard row, correctness vs torch (golden tolerance).
  test_bit_exact                   : real op vs copy on the same tensors, outputs must be bit-identical.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
import importlib

torch = importlib.import_module("torch")  # no global torch import under ttnn (pre-commit)
import ttnn

from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES
from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm

sys.path.insert(0, str(Path(__file__).parent))
from dsb_op import descriptor as copy_descriptor  # noqa: E402

entry = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile")


def _apply_knobs(monkeypatch):
    for item in filter(None, os.environ.get("LN_KNOBS", "").split(",")):
        knob, value = item.split("=")
        cast = bool if isinstance(getattr(copy_descriptor, knob), bool) else int
        monkeypatch.setattr(copy_descriptor, knob, cast(int(value)))


def _use_copy(monkeypatch):
    monkeypatch.setattr(entry, "create_program_descriptor", copy_descriptor.create_program_descriptor)
    monkeypatch.setattr(entry, "select_blocking", copy_descriptor.select_blocking)


@pytest.fixture(autouse=True)
def _variant(monkeypatch, request):
    if request.node.name.startswith("test_bit_exact"):
        return
    _apply_knobs(monkeypatch)
    if os.environ.get("LN_COPY", "1") != "0":
        _use_copy(monkeypatch)


@pytest.mark.parametrize("case", PERF_CASES, ids=IDS[: len(PERF_CASES)])
def test_perf_case(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


@pytest.mark.parametrize("case", list(GUARD_CASES.values()), ids=list(GUARD_CASES))
def test_guard_case(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


ALL_CASES = dict(zip(IDS, PERF_CASES)) | GUARD_CASES


def _capture_output(monkeypatch):
    """Wrap check_output to capture the op output as torch."""
    helpers = importlib.import_module("eval.golden_tests.layer_norm_tile.helpers")
    captured = []
    original = helpers.check_output

    def spy(ttnn_output, *args, **kwargs):
        captured.append(ttnn.to_torch(ttnn_output))
        return original(ttnn_output, *args, **kwargs)

    monkeypatch.setattr(helpers, "check_output", spy)
    return captured


@pytest.mark.parametrize("case", list(ALL_CASES.values()), ids=list(ALL_CASES))
def test_bit_exact(device, case, monkeypatch):
    case = dict(case)
    inputs = case.pop("inputs")
    captured = _capture_output(monkeypatch)
    run_layer_norm(inputs, device=device, **case)  # real op
    with monkeypatch.context() as patch:
        _apply_knobs(patch)
        _use_copy(patch)
        run_layer_norm(inputs, device=device, **case)  # copy + LN_KNOBS
    assert torch.equal(captured[0], captured[1]), "variant output differs from the real op"
