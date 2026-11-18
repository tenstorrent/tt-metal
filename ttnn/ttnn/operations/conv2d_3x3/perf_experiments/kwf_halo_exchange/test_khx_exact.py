# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision of the halo exchange: every perf focus case through the real op (HEAD 3ca4eb28f46) and through this
experiment's copy (KHX_MODE env, default 2 = exchange), same inputs, one process (not profiled). Reports bit-identity,
and both ops' PCC / mean abs error against a fp32 torch conv; asserts the focus tolerance on both.

    KHX_MODE=2 scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_halo_exchange/test_khx_exact.py
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from khx_copy import copy_modules  # noqa: E402
from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf import _FOCUS_CASES, _FOCUS_IDS  # noqa: E402
from ttnn.operations import conv2d_3x3 as real  # noqa: E402

HERE = Path(__file__).resolve().parent


def comp_pcc(a, b):
    import torch

    x = torch.stack([a.flatten().double(), b.flatten().double()])
    return None, float(torch.corrcoef(x)[0, 1])


def _run(mod, case, device, torch_input, torch_weight, torch_bias):
    import torch  # noqa: F401
    from eval.golden_tests.conv2d_3x3 import helpers

    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    memory_config = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    activation = None if case["activation"] == "none" else case["activation"]
    tt_weight = mod.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = mod.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    out = mod.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    spec = out.memory_config()
    return ttnn.to_torch(out).float(), spec


@pytest.mark.parametrize("case", _FOCUS_CASES, ids=_FOCUS_IDS)
def test_khx_exact(case, device, request):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if case["bias_mode"] == "with_bias" else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    copy_op, _ = copy_modules()
    base, base_spec = _run(real, case, device, torch_input, torch_weight, torch_bias)
    cand, cand_spec = _run(copy_op, case, device, torch_input, torch_weight, torch_bias)
    activation = None if case["activation"] == "none" else case["activation"]
    ref = helpers.pytorch_conv2d_3x3(
        torch_input,
        torch_weight,
        input_height=H,
        input_width=W,
        bias=torch_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    ).float()
    base, cand = base.reshape(ref.shape), cand.reshape(ref.shape)
    _, pcc_b = comp_pcc(ref, base)
    _, pcc_c = comp_pcc(ref, cand)
    row = {
        "id": request.node.callspec.id,
        "khx_mode": os.environ.get("KHX_MODE", "2"),
        "identical": bool(torch.equal(base, cand)),
        "differ_frac": float((base != cand).float().mean()),
        "pcc_base": float(pcc_b),
        "pcc_cand": float(pcc_c),
        "mae_base": float((base - ref).abs().mean()),
        "mae_cand": float((cand - ref).abs().mean()),
        "same_shard_spec": str(base_spec) == str(cand_spec),
    }
    with open(HERE / "precision.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row)
    assert row["same_shard_spec"]
    pcc_thr, _ = case["extras"].get("tolerance", (0.99, None))
    assert pcc_c >= pcc_thr and pcc_b >= pcc_thr


from tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf import (
    HEIGHT_SHAPES,
    _row_major_cores,
)  # noqa: E402


@pytest.mark.parametrize(
    "layer,case,dtype,layout,shard_shape,num_cores", HEIGHT_SHAPES, ids=[c[0] for c in HEIGHT_SHAPES]
)
def test_khx_exact_height_guard(layer, case, dtype, layout, shard_shape, num_cores, device):
    """The perf harness's 14 HEIGHT guard shapes (default compute config: HiFi4), real op vs copy, same inputs."""
    import torch

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride = case
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_row_major_cores(num_cores, 11), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=1,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    copy_op, _ = copy_modules()

    def run(mod):
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        out = mod.conv2d_3x3(
            tt_input,
            mod.prepare_conv2d_weights(torch_weight, **kwargs),
            input_height=H,
            input_width=W,
            bias=mod.prepare_conv2d_bias(torch_bias, **kwargs),
            padding=padding,
            stride=stride,
            activation="relu",
        )
        return ttnn.to_torch(out).float(), str(out.memory_config())

    base, base_mc = run(real)
    cand, cand_mc = run(copy_op)
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(
        torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), stride=stride, padding=padding)
    )
    ref = ref.permute(0, 2, 3, 1).reshape(base.shape)
    cand = cand.reshape(base.shape)
    _, pcc_b = comp_pcc(ref, base)
    _, pcc_c = comp_pcc(ref, cand)
    row = {
        "id": "H:" + layer,
        "khx_mode": os.environ.get("KHX_MODE", "2"),
        "identical": bool(torch.equal(base, cand)),
        "differ_frac": float((base != cand).float().mean()),
        "pcc_base": float(pcc_b),
        "pcc_cand": float(pcc_c),
        "mae_base": float((base - ref).abs().mean()),
        "mae_cand": float((cand - ref).abs().mean()),
        "same_shard_spec": base_mc == cand_mc,
    }
    with open(HERE / "precision.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    assert row["same_shard_spec"]
    assert pcc_c >= 0.99 and pcc_b >= 0.99
