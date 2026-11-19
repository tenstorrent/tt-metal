# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness of this experiment's copy (op_copy/, with its own prepare_conv2d_weights: ND-sharded HEIGHT weights,
plus whatever HWR_* switches are set) against the op at HEAD (ttnn.operations.conv2d_3x3, its own prepare: DRAM
interleaved) on every [perf] focus case, with the case's activation and without, plus PCC of each against the torch
reference. No profiler: both ops run in one process.

    HWR_SPLIT=... scripts/run_safe_pytest.sh --run-all --import-mode=prepend \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/height_weight_rate/test_hwr_precision.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hwr_copy import copy_modules  # noqa: E402

OUT = Path(__file__).resolve().parent / "precision.jsonl"


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out, ids = [], []
    for c in LOOSE_CASES:
        layer = c["extras"]["layer"]
        if "[perf]" in layer:
            out.append(c)
            ids.append(layer.split(" @")[0].split(" (")[0].replace(" ", "_").replace(".", "_").replace("+", "and"))
    return out, ids


_CASES, _IDS = _cases()


@pytest.mark.parametrize("act", ["case", "none"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_hwr_precision(case, act, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers
    import ttnn.operations.conv2d_3x3 as base

    op_mod, _ = copy_modules()
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    tw = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    has_bias = case["bias_mode"] == "with_bias"
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    activation = None if (act == "none" or case["activation"] == "none") else case["activation"]
    ref = helpers.pytorch_conv2d_3x3(
        tx,
        tw,
        input_height=H,
        input_width=W,
        bias=tb,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    ).float()
    outs, mcs, wmc = {}, {}, {}
    for name, mod in (("base", base), ("copy", op_mod)):
        x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
        w = mod.prepare_conv2d_weights(tw, **kwargs)
        wmc[name] = str(w.memory_config().nd_shard_spec is not None)
        y = mod.conv2d_3x3(
            x,
            w,
            input_height=H,
            input_width=W,
            bias=mod.prepare_conv2d_bias(tb, **kwargs) if has_bias else None,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=activation,
            compute_kernel_config=helpers.make_compute_kernel_config(
                case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
            ),
        )
        mcs[name] = y.memory_config()
        outs[name] = ttnn.to_torch(y).float().reshape(ref.shape)
    assert mcs["copy"] == mcs["base"], "the output must keep its shard spec"
    rec = {"layer": extras["layer"], "activation": activation, "nd_weights": wmc["copy"]}
    rec["env"] = {k: v for k, v in os.environ.items() if k.startswith("HWR_")}
    for name, o in outs.items():
        a, b = o.flatten(), ref.flatten()
        rec[name + "_pcc"] = round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 6)
    rec["bit_identical"] = bool(torch.equal(outs["base"], outs["copy"]))
    rec["diff_elems"] = int((outs["base"] != outs["copy"]).sum().item())
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert rec["bit_identical"], rec
