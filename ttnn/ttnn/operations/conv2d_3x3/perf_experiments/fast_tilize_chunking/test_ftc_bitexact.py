# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact gate: the op copy (lean BH fast-tilize pack) against the real conv2d_3x3 at 2ac42333eb2, same inputs,
same process (no profiler), outputs compared with torch.equal. Cases: the six [perf] focus cases, the 14 HEIGHT and
9 BLOCK Panoptic layers of the perf harness, and the DRAM-interleaved perf shapes in all three layouts.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fast_tilize_chunking/test_ftc_bitexact.py
"""

import importlib
import sys
from pathlib import Path

import pytest

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ftc_copy import copy_modules  # noqa: E402

_PERF = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf")


def _cases():
    cases = []
    for case, cid in zip(_PERF._FOCUS_CASES, _PERF._FOCUS_IDS):
        cases.append(pytest.param(("focus", case), id=f"focus-{cid}"))
    for row in _PERF.HEIGHT_SHAPES:
        cases.append(pytest.param(("height", row), id=f"height-{row[0]}"))
    for row in _PERF.BLOCK_SHAPES:
        cases.append(pytest.param(("block", row), id=f"block-{row[0]}"))
    for shape in _PERF.PERF_SHAPES:
        for layout_id, dtype, layout in (
            ("rm_bf16", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            ("tile_bf16", ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("tile_bf8b", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
        ):
            cases.append(
                pytest.param(("dram", (shape, dtype, layout)), id=f"dram-{_PERF._shape_id(shape)}-{layout_id}")
            )
    return cases


def _spec(kind, payload, device):
    """-> dict(H, W, c_in, c_out, padding, stride, dilation, dtype, layout, weight_dtype, memory_config, bias,
    activation, compute_kernel_config) or None (skip)."""
    grid = device.compute_with_storage_grid_size()
    if kind == "focus":
        from eval.golden_tests.conv2d_3x3 import helpers

        (H, W, c_in), c_out, padding, stride, dilation = payload["inputs"]
        extras = payload["extras"]
        mc = helpers.build_input_memory_config(
            (1, 1, H * W, c_in),
            payload["memory_layout"],
            layout=payload["layout"],
            dtype=payload["dtype"],
            device=device,
            extras=extras,
        )
        if mc.shard_spec is None or list(mc.shard_spec.shape) != list(extras["shard_shape"]):
            return None
        return dict(
            H=H,
            W=W,
            c_in=c_in,
            c_out=c_out,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dtype=payload["dtype"],
            layout=payload["layout"],
            weight_dtype=payload["weight_dtype"],
            memory_config=mc,
            bias=payload["bias_mode"] == "with_bias",
            activation=None if payload["activation"] == "none" else payload["activation"],
            ckc=helpers.make_compute_kernel_config(
                payload["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
            ),
        )
    if kind == "height":
        layer, ((H, W, c_in), c_out, padding, stride), dtype, layout, shard_shape, num_cores = payload
        if num_cores > grid.x * grid.y or grid.x < 11:
            return None
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_PERF._row_major_cores(num_cores, 11), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        return dict(
            H=H,
            W=W,
            c_in=c_in,
            c_out=c_out,
            padding=padding,
            stride=stride,
            dilation=1,
            dtype=dtype,
            layout=layout,
            weight_dtype=ttnn.bfloat8_b,
            memory_config=mc,
            bias=True,
            activation="relu",
            ckc=None,
        )
    if kind == "block":
        layer, ((H, W, c_in), c_out, padding, dilation), shard_shape, (gx, gy), bias_relu, fidelity, stride = payload
        if gx > grid.x or gy > grid.y:
            return None
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        return dict(
            H=H,
            W=W,
            c_in=c_in,
            c_out=c_out,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            weight_dtype=ttnn.bfloat8_b,
            memory_config=mc,
            bias=bias_relu,
            activation="relu" if bias_relu else None,
            ckc=ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False),
        )
    ((H, W, c_in), c_out, padding), dtype, layout = payload
    return dict(
        H=H,
        W=W,
        c_in=c_in,
        c_out=c_out,
        padding=padding,
        stride=1,
        dilation=1,
        dtype=dtype,
        layout=layout,
        weight_dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        bias=True,
        activation=None,
        ckc=None,
    )


def _run(op_module, s, x, w, b, device):
    import torch

    kwargs = dict(
        stride=s["stride"],
        padding=s["padding"],
        dilation=s["dilation"],
        input_dtype=s["dtype"],
        input_layout=s["layout"],
        weight_dtype=s["weight_dtype"],
        input_memory_config=s["memory_config"],
        device=device,
    )
    tt_w = op_module.prepare_conv2d_weights(w, **kwargs)
    tt_b = op_module.prepare_conv2d_bias(b, **kwargs) if b is not None else None
    tt_x = ttnn.from_torch(x, dtype=s["dtype"], layout=s["layout"], device=device, memory_config=s["memory_config"])
    extra = {} if s["ckc"] is None else dict(compute_kernel_config=s["ckc"])
    out = op_module.conv2d_3x3(
        tt_x,
        tt_w,
        input_height=s["H"],
        input_width=s["W"],
        bias=tt_b,
        padding=s["padding"],
        stride=s["stride"],
        dilation=s["dilation"],
        activation=s["activation"],
        **extra,
    )
    got = ttnn.to_torch(out).float()
    for t in (tt_x, out):
        t.deallocate()
    return got if isinstance(got, torch.Tensor) else None


@pytest.mark.parametrize("case", _cases())
def test_lean_tilize_bitexact(case, device):
    import torch

    kind, payload = case
    s = _spec(kind, payload, device)
    if s is None:
        pytest.skip("pinned grid does not fit this device")
    real = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    lean, _ = copy_modules()
    torch.manual_seed(0)
    x = torch.randn(1, 1, s["H"] * s["W"], s["c_in"]).to(torch.bfloat16)
    w = torch.randn(s["c_out"], s["c_in"], 3, 3).to(torch.bfloat16)
    b = torch.randn(s["c_out"]).to(torch.bfloat16) if s["bias"] else None
    ref = _run(real, s, x, w, b, device)
    got = _run(lean, s, x, w, b, device)
    assert torch.isfinite(ref).all()
    if not torch.equal(got, ref):
        bad = (got != ref).nonzero()
        pytest.fail(f"{bad.shape[0]} of {ref.numel()} outputs differ; first at {bad[0].tolist()}")
