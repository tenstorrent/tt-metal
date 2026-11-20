# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fill_noc_split correctness sweep: the proto with FILL_NOC_SPLIT=1 (two-NoC fill) must be bit-identical to the proto
with FILL_NOC_SPLIT=0 (the op as of this round) on interleaved ROW_MAJOR shapes beyond the FLUX focus set: aligned and
overlapping bands, dilation, valid / same / asymmetric padding, bias / no bias / ReLU, few cores, ragged blocks, odd W.
Prints whether the split engaged (the descriptor's gate) for each shape."""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

# (H, W, C_in, C_out, padding, dilation, bias, relu)
SHAPES = [
    (32, 64, 128, 128, 1, 1, True, False),
    (64, 128, 128, 128, 1, 1, True, False),
    (128, 256, 64, 64, 1, 1, True, False),
    (64, 128, 64, 64, 2, 2, True, False),
    (66, 130, 32, 64, 0, 1, True, False),
    (66, 128, 32, 32, (0, 0, 1, 1), 1, False, False),
    (16, 16, 64, 128, 1, 1, True, True),
    (130, 256, 32, 64, 1, 1, True, False),
    (258, 512, 64, 64, (0, 0, 1, 1), 1, True, False),
    (96, 200, 96, 160, 1, 1, False, True),
]


def _proto():
    name = "fns_proto"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(HERE / "proto")]
        sys.modules[name] = pkg
    return (
        importlib.import_module(f"{name}.conv2d_3x3"),
        importlib.import_module(f"{name}.conv2d_3x3_program_descriptor"),
    )


def _in_collect_pass():
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{s[0]}x{s[1]}x{s[2]}-{s[3]}-p{s[4]}-d{s[5]}" for s in SHAPES])
def test_fns_correct(device, shape):
    import torch
    import ttnn

    op, pd = _proto()
    H, W, c_in, c_out, pad, dil, has_bias, relu = shape
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    prep = dict(
        stride=1,
        padding=pad,
        dilation=dil,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=tt_in.memory_config(),
        device=device,
    )
    tt_w = op.prepare_conv2d_weights(weight, **prep)
    tt_b = op.prepare_conv2d_bias(bias, **prep) if has_bias else None
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    outs = {}
    engaged = None
    for v in ("0", "1"):
        os.environ["FILL_NOC_SPLIT"] = v
        out = op.conv2d_3x3(
            tt_in,
            tt_w,
            input_height=H,
            input_width=W,
            bias=tt_b,
            padding=pad,
            stride=1,
            dilation=dil,
            activation="relu" if relu else None,
            compute_kernel_config=cfg,
        )
        if v == "1":
            engaged = pd.FNS_LAST
        outs[v] = ttnn.to_torch(out)
        ttnn.deallocate(out)
    os.environ["FILL_NOC_SPLIT"] = "0"
    if _in_collect_pass():
        return
    n_diff = int((outs["0"] != outs["1"]).sum().item())
    print(f"FNS_CORRECT {shape}: engaged={engaged} n_diff={n_diff}")
    assert n_diff == 0, f"{shape}: split differs from base in {n_diff} elements"
