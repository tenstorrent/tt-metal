# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision of one op copy (KWCR_COPY=op_base | op_cand) on the perf focus cases (KWCR_CASES filter): PCC, mean
and max abs error against a fp32 torch conv, and the raw output saved to runs/out_<copy>_<id>.pt for a bit-level diff
across copies (compare_outputs.py).

    KWCR_COPY=op_cand scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_column_rounds/test_kwcr_precision.py"
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kwcr_copy import copy_modules, focus_cases  # noqa: E402

COPY = os.environ.get("KWCR_COPY", "op_cand")
_op, _pd = copy_modules(COPY)
_CASES, _IDS = focus_cases(os.environ.get("KWCR_CASES", ""))


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_kwcr_precision(case, device, request):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    memory_config = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip("pinned shard geometry does not fit this device's grid")
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
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    has_bias = case["bias_mode"] == "with_bias"
    activation = None if case["activation"] == "none" else case["activation"]
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tt_weight = _op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = _op.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    tt_output = _op.conv2d_3x3(
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
    got = ttnn.to_torch(tt_output).float().reshape(1, 1, -1, c_out)
    # fp32 reference on the same (host-quantized) operands the device saw.
    x_q = ttnn.to_torch(tt_input).float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x_q,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if activation == "relu":
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    if got.abs().max().item() == 0:
        return  # the up-front compile pass (no real device run)
    err = (got.double() - ref.double()).abs()
    pcc = torch.corrcoef(torch.stack([got.double().flatten(), ref.double().flatten()]))[0, 1].item()
    case_id = request.node.callspec.id
    torch.save(got, HERE / "runs" / f"out_{COPY}_{case_id}.pt")
    rel_rms = (err.pow(2).mean().sqrt() / ref.double().std()).item()
    rec = dict(copy=COPY, case=case_id, pcc=pcc, mae=err.mean().item(), max_err=err.max().item(), rel_rms=rel_rms)
    with open(HERE / "runs" / "precision.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("PRECISION", json.dumps(rec))
    tol_pcc, _ = extras.get("tolerance", (0.99, None))
    assert pcc >= tol_pcc
