# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fpu_tap_shift merge gate: the graduation copy (grad_copy/) vs the committed op on the BLOCK unit-test shapes
(tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py SHAPES x LAYOUTS, with and without bias +
ReLU, HiFi4 default config), bit for bit, in one process; plus a program-cache re-run of each copy."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fts_copy import copy_modules  # noqa: E402

UT = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded")
PARAMS = []
for case in UT.SHAPES:
    for lp in UT.LAYOUTS:
        dtype, layout = lp.values
        for bias in (True, False):
            PARAMS.append(
                pytest.param(
                    case, dtype, layout, bias, id=f"{UT._case_id(case)}-{lp.id}-{'bias_relu' if bias else 'plain'}"
                )
            )


@pytest.mark.parametrize("case,dtype,layout,bias", PARAMS)
def test_block_exact(case, dtype, layout, bias, device):
    import torch
    import ttnn

    case_hw, shards = case
    (H, W, c_in), c_out, p, s, d = case_hw
    mc = UT._block_config(H * W, c_in, layout, shards[layout])
    torch.manual_seed(1)
    w = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    b = torch.randn(c_out).to(torch.bfloat16) if bias else None
    x = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    copy_op, _ = copy_modules()
    real_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    took = {}
    real_create = copy_op.create_program_descriptor

    def spy(*args, **kwargs):
        desc = real_create(*args, **kwargs)
        took["tap_shift"] = any(any(dd[0] == "CONV2D_3X3_TAP_SHIFT" for dd in (k.defines or [])) for k in desc.kernels)
        return desc

    copy_op.create_program_descriptor = spy
    outs = {}
    try:
        for which, op_mod in (("copy", copy_op), ("copy2", copy_op), ("base", real_op)):
            kw = dict(
                stride=s,
                padding=p,
                dilation=d,
                input_dtype=dtype,
                input_layout=layout,
                weight_dtype=ttnn.bfloat8_b,
                input_memory_config=mc,
                device=device,
            )
            tt_w = op_mod.prepare_conv2d_weights(w, **kw)
            tt_b = op_mod.prepare_conv2d_bias(b, **kw) if bias else None
            tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
            y = op_mod.conv2d_3x3(
                tt_x,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b,
                padding=p,
                stride=s,
                dilation=d,
                activation="relu" if bias else None,
            )
            outs[which] = ttnn.to_torch(y).float()
            ttnn.deallocate(y)
            ttnn.deallocate(tt_x)
    finally:
        copy_op.create_program_descriptor = real_create
    diff = int((outs["copy"] != outs["base"]).sum())
    diff2 = int((outs["copy2"] != outs["base"]).sum())
    print(
        "BLOCK_EXACT", UT._case_id(case), dtype, layout, bias, "tap_shift", took.get("tap_shift"), "differ", diff, diff2
    )
    assert diff == 0 and diff2 == 0
