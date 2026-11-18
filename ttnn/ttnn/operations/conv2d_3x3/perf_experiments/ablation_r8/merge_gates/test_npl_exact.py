# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness of the narrow-planes copy (op_copy/) against the unchanged base (op_base/, = 3ca4eb28f46) on every
kw-fold [perf] focus case, with and without ReLU, plus PCC / mean abs error of each against a fp32 torch conv.
No profiler (both ops run in one process). Results append to exact.jsonl.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_narrow_planes/test_npl_exact.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npl_copy import copy_modules  # noqa: E402

KWF_LAYERS = {
    "stem.conv2": "stem_conv2",
    "stem.conv3": "stem_conv3",
    "res2.{0,1,2}.conv2": "res2",
    "res3.{1,2,3}.conv2": "res3_x",
    "instance decoder.res3.fuse_conv.0": "instance0",
    "instance decoder.res2.fuse_conv.0": "instance1",
    "instance res2.fuse_conv.1": "instance2",
    "center_head.1": "center_head_1",
}
OUT = Path(__file__).resolve().parent / "exact.jsonl"


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out, ids = [], []
    for c in LOOSE_CASES:
        layer = c["extras"]["layer"]
        for k, tag in KWF_LAYERS.items():
            if "[perf]" in layer and layer.startswith(k):
                out.append(c)
                ids.append(tag)
    return out, ids


_CASES, _IDS = _cases()


@pytest.mark.parametrize("relu", [True, False], ids=["relu", "norelu"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_npl_exact(case, relu, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    base_mod, base_pd = copy_modules("base")
    copy_mod, copy_pd = copy_modules("copy")
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
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16)
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    act = "relu" if relu else None
    xq = ttnn.to_torch(ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    wq = ttnn.to_torch(ttnn.from_torch(tw, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()
    bq = ttnn.to_torch(ttnn.from_torch(tb.reshape(1, c_out), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()[0]
    xn = xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(xn, wq, bias=bq, stride=stride, padding=padding, dilation=dilation)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    outs, kwf = {}, {}
    for name, mod, pd in (("base", base_mod, base_pd), ("copy", copy_mod, copy_pd)):
        taken = []
        orig = pd.kw_fold_mode

        def spy(*a, _orig=orig, _taken=taken, **k):
            r = _orig(*a, **k)
            _taken.append(r)
            return r

        pd.kw_fold_mode = spy
        try:
            x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
            y = mod.conv2d_3x3(
                x,
                ttnn.operations.conv2d_3x3.prepare_conv2d_weights(tw, **kwargs),
                input_height=H,
                input_width=W,
                bias=ttnn.operations.conv2d_3x3.prepare_conv2d_bias(tb, **kwargs),
                padding=padding,
                stride=stride,
                dilation=dilation,
                activation=act,
                compute_kernel_config=helpers.make_compute_kernel_config(
                    case["fp32_dest_acc_en"], extras["math_fidelity"]
                ),
            )
        finally:
            pd.kw_fold_mode = orig
        kwf[name] = any(taken)
        outs[name] = ttnn.to_torch(y).float().reshape(ref.shape)
    rec = {"layer": extras["layer"], "relu": relu, "kw_fold": kwf, "NPL_SB": os.environ.get("NPL_SB", "41")}
    for name, o in outs.items():
        a, b = o.flatten(), ref.flatten()
        rec[name] = {
            "pcc": round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 6),
            "mae": round((a - b).abs().mean().item(), 5),
        }
    d = (outs["copy"] - outs["base"]).reshape(-1, c_out)
    rec["identical_frac"] = round((d == 0).float().mean().item(), 6)
    rec["max_base_copy_diff"] = round(d.abs().max().item(), 5)
    if rec["identical_frac"] < 1:
        bad = (d != 0).nonzero()
        rec["first_bad"] = bad[:6].tolist()
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    assert kwf["base"] and kwf["copy"], rec
    assert rec["identical_frac"] == 1.0, rec
