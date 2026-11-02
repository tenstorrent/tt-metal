# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op A/B: baseline = the op's kernels; lagN = this dir's compute (per-row last K-block + epilogue,
N rows of matmul ahead). Rows: EO_ROWS="8x512,4x512"; variants: EO_VARIANTS="baseline,lag0,lag1".
Run with scripts/run_safe_pytest.sh --profile; one op per test, CSV rows in test order."""

import os
import sys
from pathlib import Path

import pytest

import importlib

op_mod = importlib.import_module("ttnn.operations.qkv_proj_heads_rope.qkv_proj_heads_rope")
import ttnn.operations.qkv_proj_heads_rope.qkv_proj_heads_rope_program_descriptor as pd
from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope


from tests.ttnn.unit_tests.operations.qkv_proj_heads_rope.test_qkv_proj_heads_rope import (
    check_outputs,
    device_inputs,
    make_inputs,
    reference,
    pcc,
    relative_rms,
)  # noqa

import ttnn

HERE_KERNELS = Path(__file__).parent / "kernels"
ROWS = [tuple(int(v) for v in r.split("x")) for r in os.environ.get("EO_ROWS", "8x512").split(",")]
VARIANTS = os.environ.get("EO_VARIANTS", "baseline,lag0,lag1").split(",")
HEADS = [int(h) for h in os.environ.get("EO_HEADS", "0").split(",")]


def _patched(lag):
    orig = pd.create_program_descriptor

    def create(*args, **kwargs):
        desc = orig(*args, **kwargs)
        kernels = list(desc.kernels)
        for kernel in kernels:
            src = Path(kernel.kernel_source)
            kernel.kernel_source = str(HERE_KERNELS / src.name)
            if src.name.endswith("_compute.cpp"):
                kernel.compile_time_args = list(kernel.compile_time_args) + [lag]
        desc.kernels = kernels
        return desc

    return create


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("batch, seq", ROWS, ids=[f"{b}x{s}" for b, s in ROWS])
def test_ab(device, monkeypatch, batch, seq, heads, variant):
    if heads:  # 0 = the selection function's own pick
        monkeypatch.setattr(pd, "HEADS_PER_CORE_CANDIDATES", (heads,))
    if variant != "baseline":
        monkeypatch.setattr(op_mod, "create_program_descriptor", _patched(int(variant.removeprefix("lag"))))
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    tensors = device_inputs(device, x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(*tensors)
    expected = reference(x, w, b, cos, sin)
    for name, out, ref in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out)
        print(
            f"EO_PREC {batch}x{seq} h{heads} {variant} {name} pcc={pcc(actual, ref):.6f} rms={relative_rms(actual, ref):.3e}"
        )
    check_outputs(outputs, expected)
