# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact gate: op_copy/ (with the KRN_* switches of the environment) vs op_base/ (HEAD 039e18f6c81) on the [perf]
focus cases, one process. Prints differing elements and the max abs difference; every case must be bit-identical (the
variants only move data-movement timing / NoCs). KRN_EXACT_COPY=grad compares grad/ (HEAD + graduation.patch)."""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _modules(pkg_name, sub):
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    return importlib.import_module(f"{pkg_name}.conv2d_3x3")


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cs = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    return cs, [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in cs]


CASES, IDS = _cases()


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_krn_exact(case, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, layout = case["dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    if mc.shard_spec is None or list(mc.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip("pinned shard geometry does not fit this device's grid")
    has_bias = case["bias_mode"] == "with_bias"
    act = None if case["activation"] == "none" else case["activation"]
    torch.manual_seed(0)
    w = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    b = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    x = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    copy_sub = os.environ.get("KRN_EXACT_COPY", "op_copy")
    outs = {}
    for which, op_mod in (("copy", _modules("krn_x_copy", copy_sub)), ("base", _modules("krn_x_base", "op_base"))):
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
    ref = helpers.pytorch_conv2d_3x3(
        x, w, input_height=H, input_width=W, bias=b, padding=padding, stride=stride, dilation=dilation, activation=act
    ).float()
    got = outs["copy"].reshape(ref.shape)
    pcc = float(torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1])
    diff = int((outs["copy"] != outs["base"]).sum())
    maxd = float((outs["copy"] - outs["base"]).abs().max())
    env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("KRN_"))
    print("KRN_EXACT", env, extras["layer"], "differ", diff, "of", outs["base"].numel(), "max", maxd, "pcc", pcc)
    assert diff == 0
