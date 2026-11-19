# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness of the fused side-rounds copy (op_copy/) against HEAD (op_base/) in one process (no profiler):
- every kw-fold [perf] focus case, with and without ReLU, plus PCC / mean abs error of each against a fp32 torch conv
  of the bf8_b-quantized operands;
- the forced-kw-fold unit shapes and the halo-exchange unit shapes (tests/.../test_conv2d_3x3_kw_fold.py), kw-fold
  forced on both copies, the exchange on (one M-block per core) or recompute (more).
Results append to exact.jsonl.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_fused_side_rounds/test_fsr_exact.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsr_copy import copy_modules  # noqa: E402

KWF_LAYERS = {
    "stem.conv2": "stem_conv2",
    "stem.conv3": "stem_conv3",
    "res2.{0,1,2}.conv2": "res2",
    "res3.{1,2,3}.conv2": "res3_x",
    "semantic decoder.res3.fuse_conv.0": "sem_dec_res3_fc0",
    "instance decoder.res3.fuse_conv.0": "inst_dec_res3_fc0",
    "semantic decoder.res3.fuse_conv.1": "sem_dec_res3_fc1",
    "semantic decoder.res2.fuse_conv.0": "sem_dec_res2_fc0",
    "instance decoder.res2.fuse_conv.0": "inst_dec_res2_fc0",
    "semantic res2.fuse_conv.1": "sem_res2_fc1",
    "instance res2.fuse_conv.1": "inst_res2_fc1",
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


def _run_both(build_input, conv_kwargs, prep_kwargs, tw, tb, force=None):
    import ttnn

    outs, kwf = {}, {}
    for name in ("base", "copy"):
        mod, pd = copy_modules(name)
        taken = []
        saved = {k: getattr(pd, k) for k in ("kw_fold_mode", "_kw_fold_pays", "kw_fold_halo_exchange")}
        orig = pd.kw_fold_mode

        def spy(*a, _orig=orig, _taken=taken, **k):
            r = _orig(*a, **k)
            _taken.append(r)
            return r

        pd.kw_fold_mode = spy
        if force is not None:
            pd._kw_fold_pays = lambda *a: True
            pd.kw_fold_halo_exchange = lambda b, _x=force: _x and b.max_m_blocks_per_core == 1
        try:
            y = mod.conv2d_3x3(
                build_input(),
                ttnn.operations.conv2d_3x3.prepare_conv2d_weights(tw, **prep_kwargs),
                bias=ttnn.operations.conv2d_3x3.prepare_conv2d_bias(tb, **prep_kwargs),
                **conv_kwargs,
            )
        finally:
            for k, v in saved.items():
                setattr(pd, k, v)
        kwf[name] = any(taken)
        outs[name] = ttnn.to_torch(y).float()
    return outs, kwf


def _record(rec, outs, ref, c_out):
    import torch

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
        rec["first_bad"] = (d != 0).nonzero()[:6].tolist()
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


@pytest.mark.parametrize("relu", [True, False], ids=["relu", "norelu"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_fsr_exact_focus(case, relu, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    prep = dict(
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
    xq = ttnn.to_torch(ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    wq = ttnn.to_torch(ttnn.from_torch(tw, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()
    bq = ttnn.to_torch(ttnn.from_torch(tb.reshape(1, c_out), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()[0]
    ref = torch.nn.functional.conv2d(
        xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2), wq, bias=bq, stride=stride, padding=padding, dilation=dilation
    )
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    conv = dict(
        input_height=H,
        input_width=W,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if relu else None,
        compute_kernel_config=helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras["math_fidelity"]),
    )
    outs, kwf = _run_both(
        lambda: ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc), conv, prep, tw, tb
    )
    outs = {k: v.reshape(ref.shape) for k, v in outs.items()}
    rec = {"case": extras["layer"], "relu": relu, "kw_fold": kwf}
    _record(rec, outs, ref, c_out)
    assert kwf["base"] and kwf["copy"], rec
    assert rec["identical_frac"] == 1.0, rec


def _unit_cases():
    from tests.ttnn.unit_tests.operations.conv2d_3x3 import test_conv2d_3x3_kw_fold as u

    # Extra 2-tile n-group shapes the unit files do not reach: K-blocked N = 2, several n-groups (N = 10: whole-K and
    # K-blocked), a one-tile-row core, the exchange off (several M-blocks) on K-blocked N = 2.
    extra = [
        ("kblocked_n2", 32, 64, 256, 64, 1, "bf8", "bf8", "LoFi", True, 16),
        ("kblocked_n2_hifi4", 64, 128, 256, 64, 1, "bf8", "bf8", "HiFi4", False, 86),
        ("n10_whole_k", 32, 64, 32, 320, 1, "bf8", "bf8", "LoFi", True, 16),
        ("n10_kblocked", 32, 128, 64, 320, 1, "bf8", "bf8", "LoFi", True, 32),
        ("n2_one_row_cores", 32, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 64),
        ("n2_d3", 48, 96, 64, 64, 3, "bf16", "bf8", "LoFi", True, 36),
        ("kblocked_n2_multiblock", 64, 128, 256, 64, 1, "bf8", "bf8", "LoFi", True, 8),
    ]
    cases = [("forced", c) for c in u.CASES] + [("xchg", c) for c in u.XCHG_CASES] + [("extra", c) for c in extra]
    return cases, [f"{k}-{c[0]}" for k, c in cases]


_UNIT, _UNIT_IDS = _unit_cases()


@pytest.mark.parametrize("kind,case", _UNIT, ids=_UNIT_IDS)
def test_fsr_exact_unit(kind, case, device):
    import torch
    import ttnn

    _, H, W, c_in, c_out, d, act, wdt, fidelity, relu, cores = case
    DT = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    dtype, weight_dtype = DT[act], DT[wdt]
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    if cores > grid.x * grid.y:
        pytest.skip("grid")
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    torch.manual_seed(0)
    tx = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tw = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    tb = torch.randn(c_out).to(torch.bfloat16)
    prep = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    conv = dict(
        input_height=H,
        input_width=W,
        padding=d,
        stride=1,
        dilation=d,
        activation="relu" if relu else None,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True
        ),
    )
    x = tx.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, tw.float(), tb.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    outs, kwf = _run_both(
        lambda: ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc),
        conv,
        prep,
        tw,
        tb,
        force=True,
    )
    outs = {k: v.reshape(ref.shape) for k, v in outs.items()}
    rec = {"case": f"{kind}-{case[0]}", "kw_fold": kwf}
    _record(rec, outs, ref, c_out)
    assert kwf["base"] and kwf["copy"], rec
    assert rec["identical_frac"] == 1.0, rec
