# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision of the seedless kw-fold copy vs the base op (ttnn.operations.conv2d_3x3 at e40c435b625), on the kw-fold
[perf] focus cases: PCC and mean abs error of each against a fp32 torch conv, the share of bit-identical outputs, and
the largest base-vs-copy difference. No profiler (both ops run in one process).

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_seedless/test_kwfs_precision.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwfs_copy import copy_modules  # noqa: E402

KWF_LAYERS = {
    "stem.conv2": "stem2",
    "stem.conv3": "stem3",
    "instance decoder.res3.fuse_conv.0": "instdec_r3_fc0",
    "instance decoder.res2.fuse_conv.0": "instdec_r2_fc0",
    "instance res2.fuse_conv.1": "inst_r2_fc1",
    "center_head.1": "center1",
}
OUT = Path(__file__).resolve().parent / "precision.jsonl"


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
def test_kwfs_precision(case, relu, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers
    from ttnn.operations.conv2d_3x3 import conv2d_3x3 as base_op, prepare_conv2d_bias, prepare_conv2d_weights

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
    if os.environ.get("KWFS_WMASK"):
        # Debug: keep only the listed kw columns of the weights.
        keep = [int(v) for v in os.environ["KWFS_WMASK"]]
        mask = torch.zeros(3, dtype=torch.bfloat16)
        mask[keep] = 1
        tw = tw * mask.reshape(1, 1, 1, 3)
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16)
    if os.environ.get("KWFS_BIAS_RAMP"):
        tb = (torch.arange(c_out) + 1).to(torch.bfloat16)  # debug: bias[c] = c + 1
        if os.environ["KWFS_BIAS_RAMP"] == "one":
            tb = torch.ones(c_out, dtype=torch.bfloat16)
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    act = "relu" if relu else None
    # fp32 reference (the helper rounds it to the input dtype); the bf8_b input / weights the device sees.
    xq = ttnn.to_torch(ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    wq = ttnn.to_torch(ttnn.from_torch(tw, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()
    bq = ttnn.to_torch(ttnn.from_torch(tb.reshape(1, c_out), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()[0]
    xn = xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(xn, wq, bias=bq, stride=stride, padding=padding, dilation=dilation)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    outs = {}
    for name, fn in (("base", base_op), ("copy", op_mod.conv2d_3x3)):
        x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
        y = fn(
            x,
            prepare_conv2d_weights(tw, **kwargs),
            input_height=H,
            input_width=W,
            bias=prepare_conv2d_bias(tb, **kwargs),
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=act,
            compute_kernel_config=helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras["math_fidelity"]),
        )
        outs[name] = ttnn.to_torch(y).float().reshape(ref.shape)
    if os.environ.get("KWFS_DIAG2"):
        # Debug: the copy's output minus the copy with the plain final pass (no bias add) = what the add contributed.
        saved = os.environ.get("KWFS_DEFINES", "")
        os.environ["KWFS_DEFINES"] = "KWFS_DBG_OLD_FINAL"
        x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
        y = op_mod.conv2d_3x3(
            x,
            prepare_conv2d_weights(tw, **kwargs),
            input_height=H,
            input_width=W,
            bias=prepare_conv2d_bias(tb, **kwargs),
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=act,
            compute_kernel_config=helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras["math_fidelity"]),
        )
        os.environ["KWFS_DEFINES"] = saved
        plain = ttnn.to_torch(y).float().reshape(-1, c_out)
        added = outs["copy"].reshape(-1, c_out) - plain
        torch.set_printoptions(linewidth=300, precision=2, sci_mode=False, threshold=100000)
        print("KWFS_ADDED tile(0,0) rows 0..31 cols 0..31\n", added[:32, :32])
        print("KWFS_BIAS", bq[:32])
        print("KWFS_ADDED mean over tile-rows\n", added.reshape(-1, 32, c_out).mean(0)[:, :32])
    rec = {"layer": extras["layer"], "relu": relu}
    for name, o in outs.items():
        a, b = o.flatten(), ref.flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        rec[name] = {
            "pcc": round(pcc, 6),
            "mae": round((a - b).abs().mean().item(), 5),
            "maxe": round((a - b).abs().max().item(), 4),
        }
    d = (outs["copy"] - outs["base"]).reshape(-1, c_out)
    rec["identical_frac"] = round((d == 0).float().mean().item(), 6)
    rec["max_base_copy_diff"] = round(d.abs().max().item(), 5)
    if os.environ.get("KWFS_DIAG"):
        # Where the copy differs: per output column, per row-in-tile, per tile-row (first rows of the op's output).
        rows = d.shape[0]
        rec["diff_col_mean"] = [round(v, 3) for v in d.mean(0)[:32].tolist()]
        rec["diff_rowmod32_absmean"] = [
            round(v, 3) for v in d.abs().reshape(rows // 32, 32, c_out).mean((0, 2)).tolist()
        ]
        bias_b = bq
        rec["bias_head"] = [round(v, 3) for v in bias_b[:8].tolist()]
        rec["copy_minus_ref_col_mean"] = [
            round(v, 3) for v in (outs["copy"] - ref).reshape(-1, c_out).mean(0)[:8].tolist()
        ]
    print("KWFS_PREC", json.dumps(rec))
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
