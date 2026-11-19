# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""height_round0_latency bit-exact gate: this experiment's op copy (op_copy/, variant picked by HRL_* environment
variables) against the real op at HEAD (ttnn.operations.conv2d_3x3), in one process, on every [perf] focus case and the
HEIGHT perf layers. 0 differing output elements is the pass."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hrl_copy import copy_modules  # noqa: E402


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cs = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    ids = [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in cs]
    return cs, ids


CASES, IDS = _cases()


def _run(op_mod, case, device, w, b, x, mc):
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    kw = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=mc,
        device=device,
    )
    tt_w = op_mod.prepare_conv2d_weights(w, **kw)
    tt_b = op_mod.prepare_conv2d_bias(b, **kw) if b is not None else None
    tt_x = ttnn.from_torch(x, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=mc)
    act = None if case["activation"] == "none" else case["activation"]
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
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    out = ttnn.to_torch(y).float()
    ttnn.deallocate(y)
    ttnn.deallocate(tt_x)
    return out


@pytest.mark.parametrize("rep", [0, 1], ids=["run0", "rerun"])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_hrl_exact(case, rep, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers
    import importlib

    real_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")

    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in),
        case["memory_layout"],
        layout=case["layout"],
        dtype=case["dtype"],
        device=device,
        extras=extras,
    )
    if mc.shard_spec is None or list(mc.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip("pinned shard geometry does not fit this device's grid")
    torch.manual_seed(rep)
    w = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    b = helpers.make_bias(c_out, dtype=torch.bfloat16) if case["bias_mode"] == "with_bias" else None
    x = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    op_copy, _ = copy_modules()
    real = _run(real_op, case, device, w, b, x, mc)
    copy = _run(op_copy, case, device, w, b, x, mc)
    diff = int((real != copy).sum())
    print("HRL_EXACT", extras["layer"], "rep", rep, "differ", diff, "of", real.numel())
    assert diff == 0
