# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fill_shrink bake-off: the real op (baseline) vs the forked kernels with a ragged K-block ramp
(QKV_FILL_RAMP). Rows from QKV_FS_ROWS ("8x512,4x512"), variant from QKV_FS_VARIANT (real | fork).
Correctness-gated with the acceptance test's check_outputs. Run via run.sh (profiled)."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope

op_mod = importlib.import_module("ttnn.operations.qkv_proj_heads_rope.qkv_proj_heads_rope")
pd_real = importlib.import_module("ttnn.operations.qkv_proj_heads_rope.qkv_proj_heads_rope_program_descriptor")

_REPO = Path(__file__).resolve().parents[6]
_spec = importlib.util.spec_from_file_location(
    "qkv_acceptance", _REPO / "tests/ttnn/unit_tests/operations/qkv_proj_heads_rope/test_qkv_proj_heads_rope.py"
)
acc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = acc
_spec.loader.exec_module(acc)
_spec = importlib.util.spec_from_file_location("fill_shrink_pd_fork", Path(__file__).parent / "pd_fork.py")
pd_fork = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pd_fork
_spec.loader.exec_module(pd_fork)

ROWS = [tuple(int(v) for v in r.split("x")) for r in os.environ.get("QKV_FS_ROWS", "8x512").split(",")]
VARIANT = os.environ.get("QKV_FS_VARIANT", "fork")
# R1 forcing: QKV_FS_WMIN=99 disables W multicast.
WMIN = os.environ.get("QKV_FS_WMIN")


def run_fork(input_tensor, weight, bias, cos, sin):
    cfg = op_mod.default_compute_kernel_config()
    op_mod.validate(input_tensor, weight, bias, cos, sin, compute_kernel_config=cfg)
    batch, seq, head_dim = op_mod._check_shapes(input_tensor, weight, bias, cos, sin, op_mod.NUM_HEADS)
    device = input_tensor.device()
    out_shape = ttnn.Shape([batch, op_mod.NUM_HEADS, seq, head_dim])
    outputs = tuple(
        ttnn.allocate_tensor_on_device(
            out_shape, input_tensor.dtype, ttnn.TILE_LAYOUT, device, input_tensor.memory_config()
        )
        for _ in range(3)
    )
    blocking = pd_fork.select_blocking(device, batch, seq, cfg)
    print("FILL_SHRINK blocking", blocking, "ramp", pd_fork.fill_ramp())
    desc = pd_fork.create_program_descriptor(input_tensor, weight, bias, cos, sin, outputs, blocking, cfg)
    ttnn.generic_op([input_tensor, weight, bias, cos, sin, *outputs], desc)
    return outputs


@pytest.mark.parametrize("batch, seq", ROWS, ids=[f"{b}x{s}" for b, s in ROWS])
def test_fill_shrink(device, monkeypatch, batch, seq):
    if WMIN:
        monkeypatch.setattr(pd_real, "W_MCAST_MIN_GROUP", int(WMIN))
        monkeypatch.setattr(pd_fork, "W_MCAST_MIN_GROUP", int(WMIN))
    x, w, b, cos, sin = acc.make_inputs(batch, seq, seq)
    tensors = acc.device_inputs(device, x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(*tensors) if VARIANT == "real" else run_fork(*tensors)
    expected = acc.reference(x, w, b, cos, sin)
    for name, out, ref in zip("qkv", outputs, expected):
        actual = ttnn.to_torch(out)
        print(f"FILL_SHRINK pcc {name} {acc.pcc(actual, ref):.6f} rms {acc.relative_rms(actual, ref):.3e}")
    acc.check_outputs(outputs, expected)
