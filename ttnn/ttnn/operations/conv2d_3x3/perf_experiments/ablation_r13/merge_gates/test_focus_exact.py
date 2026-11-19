"""Perf 13 merge gate: the final tree (op_copy/) vs 4f2f03f6d20 (op_base/) on all 22 [perf] focus cases. Reports differing
elements and the max abs difference; every case must be bit-identical."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from op_copies import copy_modules  # noqa: E402


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES
    import ttnn

    cs = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    return cs, [c["extras"]["layer"].split(" ")[0] for c in cs]


CASES, IDS = _cases()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_focus_exact(case, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, layout = case["dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    has_bias = case["bias_mode"] == "with_bias"
    act = None if case["activation"] == "none" else case["activation"]
    torch.manual_seed(0)
    w = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    b = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    x = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    outs = {}
    for which in ("copy", "base"):
        op_mod, _ = copy_modules(which)
        kw = dict(
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_dtype=dtype,
            input_layout=layout,
            weight_dtype=case["weight_dtype"],
            input_memory_config=mc,
            device=device,
        )
        tt_w = op_mod.prepare_conv2d_weights(w, **kw)
        tt_b = op_mod.prepare_conv2d_bias(b, **kw) if has_bias else None
        tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
        y = op_mod.conv2d_3x3(
            tt_x,
            tt_w,
            input_height=H,
            input_width=W,
            bias=tt_b,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=act,
            compute_kernel_config=helpers.make_compute_kernel_config(True, extras["math_fidelity"]),
        )
        outs[which] = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
        ttnn.deallocate(tt_x)
    diff = int((outs["copy"] != outs["base"]).sum())
    maxd = float((outs["copy"] - outs["base"]).abs().max())
    print("FOCUS_EXACT", case["extras"]["layer"], "differ", diff, "of", outs["base"].numel(), "max", maxd)
    assert diff == 0
