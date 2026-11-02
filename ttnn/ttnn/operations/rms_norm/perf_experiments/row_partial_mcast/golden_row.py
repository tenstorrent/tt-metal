# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden correctness sweep of the row_partial_mcast clone (final candidate: row partial + matmul collapse + root
multicast + owner spread). Not collected by a directory run (no test_ prefix).

A stratified sample of the golden registry cases that can reach the owner combine (WIDTH / BLOCK_SHARDED, and
INTERLEAVED with W >= 288, where k > 8 is possible): every distinct (shape, Layout, TensorMemoryLayout) once, the other
axes rotated, plus every loose case of those memory layouts. Each runs through the golden harness's run_rms_norm
(check_output vs torch) with the op's create_program_descriptor swapped for the clone's.

  GR_STRIDE=n (default 1)  keep every n-th sampled case
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

from eval.golden_harness import parametrize_cases, parametrize_loose_cases
from eval.golden_tests.rms_norm.feature_spec import INPUTS, INVALID, LOOSE_CASES, TARGET
from eval.golden_tests.rms_norm.helpers import run_rms_norm
from ttnn.operations.rms_norm import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED
import ttnn.operations.rms_norm.rms_norm  # noqa: F401

HERE = Path(__file__).parent
# GR_PD=grad (default): the graduation-ready descriptor; GR_PD=clone: the experiment clone with every lever on
_PD = HERE / ("pd_row.py" if os.environ.get("GR_PD") == "clone" else "graduate/rms_norm_program_descriptor.py")
spec = importlib.util.spec_from_file_location("rms_norm_pd_row_golden", _PD)
clone = importlib.util.module_from_spec(spec)
sys.modules["rms_norm_pd_row_golden"] = clone
spec.loader.exec_module(clone)
if os.environ.get("GR_PD") == "clone":
    clone.ROW_PARTIAL = clone.INV_ROOT_MCAST = clone.ROW_COLLAPSE_MM = clone.OWNER_SPREAD = True
OPMOD = sys.modules["ttnn.operations.rms_norm.rms_norm"]
ML = ttnn.TensorMemoryLayout


def _reaches_owner(shape, axes):
    ml = axes.get("memory_layout")
    if ml in (ML.WIDTH_SHARDED, ML.BLOCK_SHARDED):
        return True
    return ml == ML.INTERLEAVED and shape[-1] >= 288


def _sample():
    seen = {}
    out = []
    for p in parametrize_cases(TARGET, INPUTS, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID):
        inputs, axes = p.values
        if not _reaches_owner(inputs[0], axes):
            continue
        key = (tuple(inputs[0]), str(axes["layout"]), str(axes["memory_layout"]))
        n = seen.get(key, 0)
        seen[key] = n + 1
        # rotate the other axes: keep the 0th, 7th, 23rd variant of each key (dtype / gamma / fp32 DEST mix)
        if n in (0, 7, 23):
            out.append(pytest.param(inputs, axes, None, id=f"c{len(out)}", marks=p.marks))
    for p in parametrize_loose_cases(LOOSE_CASES, INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, INVALID):
        inputs, axes, extras = p.values
        if _reaches_owner(inputs[0], axes):
            out.append(pytest.param(inputs, axes, extras, id=f"l{len(out)}", marks=p.marks))
    stride = int(os.environ.get("GR_STRIDE", "1"))
    return out[::stride]


CASES = _sample()


@pytest.mark.parametrize("inputs,axes,extras", CASES)
def test_golden_row(inputs, axes, extras, device, monkeypatch):
    monkeypatch.setattr(OPMOD, "create_program_descriptor", clone.create_program_descriptor)
    run_rms_norm(inputs, device=device, extras=extras, **axes)
