# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The kw-fold [perf] focus cases (exact config of test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus), run once per
variant in the sequence KFP_SEQ (comma list of op copy dirs, e.g. "op_base_unz,op_copy_unz"), case by case, in one
process. KFP_CASES (comma list of short ids, default all 12) selects cases. PCC asserted against torch fp32 per run.

    KFP_SEQ=op_base_unz,op_copy_unz scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_final_pass/test_kfp_perf.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kfp_copy import copy_modules  # noqa: E402

# short id -> feature_spec layer name (the 12 focus cases that take kw-fold)
KWF_CASES = {
    "stem_conv2": "stem.conv2",
    "stem_conv3": "stem.conv3",
    "res2": "res2.{0,1,2}.conv2",
    "res3x": "res3.{1,2,3}.conv2 + instance res3.fuse_conv.1",
    "sem_dec_res3_fc0": "semantic decoder.res3.fuse_conv.0",
    "inst_dec_res3_fc0": "instance decoder.res3.fuse_conv.0",
    "sem_dec_res3_fc1": "semantic decoder.res3.fuse_conv.1",
    "sem_dec_res2_fc0": "semantic decoder.res2.fuse_conv.0",
    "inst_dec_res2_fc0": "instance decoder.res2.fuse_conv.0",
    "sem_res2_fc1": "semantic res2.fuse_conv.1 + semantic_head.head.{0,1}",
    "inst_res2_fc1": "instance res2.fuse_conv.1 + center_head.0 + offset_head.0",
    "center_head1": "center_head.1 + offset_head.1",
}


def _params():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    by_layer = {c["extras"]["layer"]: c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]}
    want = os.environ.get("KFP_CASES", ",".join(KWF_CASES)).split(",")
    seq = os.environ.get("KFP_SEQ", "op_base_unz,op_copy_unz").split(",")
    out = []
    for cid in want:
        case = by_layer[f"{KWF_CASES[cid]} @ LoFi [perf]"]
        for k, v in enumerate(seq):
            out.append(pytest.param(case, v, id=f"{cid}-{k}-{v}"))
    return out


@pytest.mark.parametrize("case,variant", _params())
def test_kfp_perf(case, variant, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers
    from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

    op_mod, _ = copy_modules(variant)
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    kw = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    w = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    has_bias = case["bias_mode"] == "with_bias"
    act = None if case["activation"] == "none" else case["activation"]
    b = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tt_w = op_mod.prepare_conv2d_weights(w, **kw)
    tt_b = op_mod.prepare_conv2d_bias(b, **kw) if has_bias else None
    x = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
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
        compute_kernel_config=helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras["math_fidelity"]),
    )
    ttnn.ReadDeviceProfiler(device)
    ref = helpers.pytorch_conv2d_3x3(
        x,
        w,
        input_height=H,
        input_width=W,
        bias=b,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=act,
    )
    got = ttnn.to_torch(y).float().reshape(ref.shape)
    _, pcc = comp_pcc(ref.float(), got, 0.996)
    print(f"KFP_PCC {variant} {extras['layer']} pcc {pcc}")
    if os.environ.get("KFP_NOCHECK") != "1":
        assert_with_pcc(ref.float(), got, extras.get("tolerance", (0.99, None))[0])
