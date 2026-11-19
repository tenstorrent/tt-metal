# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 11 `tap_shift_rate` op-level A/B: HEAD (op_base*) vs the row-buffer tap shift (op_copy*) on the focus cases.

Variants (TSR_VARIANTS, comma list):
    base / cand          op_base_unz / op_copy_unz (MaybeDeviceZoneScope stripped: make_unz.sh), as committed
    base_nc / cand_nc    the same with the tap-shift carve-out off (TAP_SHIFT_MIN_HALO_PER_TILE = 0)
    base_z / cand_z      the zoned copies (op_base / op_copy)
TSR_CASES: comma list of focus ids (layer name, '.' -> '_', braces dropped, e.g. res4_1__5_conv2) or
`block:<name>` (the BLOCK guard shapes below). TSR_ORDER=alt reverses the variant order on every other case;
TSR_LOG names the run log (runs/<log>.jsonl): one JSON record per dispatch (variant, case, PCC, output sha), joined
with the profiler CSV by report.py. The precision contract is the case's: no variant touches it.
"""

import hashlib
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).resolve().parent
COPIES = {"base": "op_base_unz", "cand": "op_copy_unz", "base_z": "op_base", "cand_z": "op_cand"}


def _mods(sub):
    pkg_name = f"tsr_{sub}"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[pkg_name] = pkg
    op = importlib.import_module(f"{pkg_name}.conv2d_3x3")
    pd = importlib.import_module(f"{pkg_name}.conv2d_3x3_program_descriptor")
    assert Path(pd.KERNEL_DIR).parent == HERE / sub, pd.KERNEL_DIR
    return op, pd


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_").replace("{", "").replace("}", "")] = c
    return out


FOCUS = _focus_cases()
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride, dtype)
BLOCK_SHAPES = {
    "res4_conv2_hifi4": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1, "bf8"),
    "res4_conv2_bf16": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "LoFi", 1, "bf16"),
    "res5_d2_bf16": (((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, "LoFi", 1, "bf16"),
    "res5_d3": (((32, 64, 512), 512, 3, 3), [224, 64], (8, 10), True, "LoFi", 1, "bf8"),
    "aspp_d6_hifi4": (((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, "HiFi4", 1, "bf8"),
    "w128_d1": (((16, 128, 256), 256, 1, 1), [256, 32], (8, 8), True, "LoFi", 1, "bf8"),
    "w128_d5": (((32, 128, 256), 256, 5, 5), [512, 32], (8, 8), True, "LoFi", 1, "bf8"),
    "w96_d7": (((32, 96, 256), 256, 7, 7), [384, 32], (8, 8), True, "LoFi", 1, "bf8"),
}

SEL_CASES = [c for c in os.environ.get("TSR_CASES", "res4_1__5_conv2").split(",") if c]
SEL_VARIANTS = [v for v in os.environ.get("TSR_VARIANTS", "base,cand").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("TSR_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("TSR_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def _variant(variant, monkeypatch):
    nc = variant.endswith("_nc")
    key = variant[:-3] if nc else variant
    op, pd = _mods(COPIES[key])
    if nc:
        if hasattr(pd, "TAP_SHIFT_MIN_HALO_PER_TILE"):
            monkeypatch.setattr(pd, "TAP_SHIFT_MIN_HALO_PER_TILE", 0)
    pd.select_blocking.cache_clear()
    return op, pd


def _run(
    op, device, H, W, c_in, c_out, padding, dilation, stride, mc, dtype, layout, wdtype, ckc, bias, act, seed_inputs
):
    import torch

    torch.manual_seed(0)
    kw = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=wdtype,
        input_memory_config=mc,
        device=device,
    )
    tw, tb, x = seed_inputs()
    tt_w = op.prepare_conv2d_weights(tw, **kw)
    tt_b = op.prepare_conv2d_bias(tb, **kw) if bias else None
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    y = op.conv2d_3x3(
        tt_x,
        tt_w,
        input_height=H,
        input_width=W,
        bias=tt_b,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=act,
        compute_kernel_config=ckc,
    )
    return y, tw, tb, x


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_tsr(case_id, variant, device, monkeypatch):
    import torch

    op, pd = _variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    real_cpd = op.create_program_descriptor

    def recording_cpd(*args, **kwargs):
        desc = real_cpd(*args, **kwargs)
        defs = set()
        for k in desc.kernels:
            for kv in k.defines:
                defs.add(kv[0])
        info["tap_shift"] = "CONV2D_3X3_TAP_SHIFT" in defs
        return desc

    monkeypatch.setattr(op, "create_program_descriptor", recording_cpd)
    if case_id.startswith("block:"):
        name = case_id.split(":", 1)[1]
        ((H, W, c_in), c_out, padding, dilation), shard, (gx, gy), bias, fid, stride, dt = BLOCK_SHAPES[name]
        grid = device.compute_with_storage_grid_size()
        if gx > grid.x or gy > grid.y:
            pytest.skip("grid")
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
                shard,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        dtype = ttnn.bfloat8_b if dt == "bf8" else ttnn.bfloat16
        act = "relu" if bias else None

        def seed():
            tw = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
            tb = torch.randn(c_out).to(torch.bfloat16) if bias else None
            return tw, tb, torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)

        ckc = ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, math_approx_mode=False
        )
        y, tw, tb, x = _run(
            op,
            device,
            H,
            W,
            c_in,
            c_out,
            padding,
            dilation,
            stride,
            mc,
            dtype,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat8_b,
            ckc,
            bias,
            act,
            seed,
        )
        xr = x.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
        ref = torch.nn.functional.conv2d(
            xr, tw.float(), None if tb is None else tb.float(), stride=stride, padding=padding, dilation=dilation
        )
        if bias:
            ref = torch.relu(ref)
        ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
        pcc_thr = 0.99
    else:
        from eval.golden_tests.conv2d_3x3 import helpers

        case = FOCUS[case_id]
        (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
        extras = case["extras"]
        dtype, layout = case["dtype"], case["layout"]
        mc = helpers.build_input_memory_config(
            (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
        )
        if mc.shard_spec is None or list(mc.shard_spec.shape) != list(extras["shard_shape"]):
            pytest.skip("pinned shard geometry does not fit this grid")
        bias = case["bias_mode"] == "with_bias"
        act = None if case["activation"] == "none" else case["activation"]

        def seed():
            tw = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
            tb = helpers.make_bias(c_out, dtype=torch.bfloat16) if bias else None
            return tw, tb, helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)

        ckc = helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        )
        y, tw, tb, x = _run(
            op,
            device,
            H,
            W,
            c_in,
            c_out,
            padding,
            dilation,
            stride,
            mc,
            dtype,
            layout,
            case["weight_dtype"],
            ckc,
            bias,
            act,
            seed,
        )
        ref = helpers.pytorch_conv2d_3x3(
            x.float(),
            tw.float(),
            input_height=H,
            input_width=W,
            bias=None if tb is None else tb.float(),
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation=act,
        )
        pcc_thr = (extras.get("tolerance") or (0.99, None))[0]
    out = ttnn.to_torch(y).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_thr)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_thr,
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert info["finite"] and passed, info
