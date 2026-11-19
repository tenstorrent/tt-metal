# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B (perf experiment, round 12, `stride2_dest_taps`): stride-2 im2col taps built from the input's TILE slab in DEST
instead of untilize + window gather + strided tilize.

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 739a271172e, unmodified copy)
  cand      op_copy/ with S2_TAPS = True
  off       op_copy/ with S2_TAPS = False (must behave as HEAD)
  grad      grad/ (HEAD + graduation.patch)
S2_NZ=1 selects the zone-stripped twins (*_nz, mk_nz.sh) for the unzoned verdict numbers.

Env: S2_CASES (focus ids, `golden:<i>:<layout>` or `domain`; comma list), S2_VARIANTS (comma list, run in that order per
case), S2_ORDER=alt (reverse the order on every other case), S2_LOG (run-log name), S2_NOCHECK=1. Each dispatch appends
a JSON record (blocking, PCC, mean / max abs error vs a fp32 torch conv, output hash, the output .pt) to
runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp rows in call order. The precision contract is
the case's: no variant touches it. Inputs are seeded per case.
"""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.stride2_dest_taps."
_NZ = "_nz" if os.environ.get("S2_NZ") == "1" else ""
_PKGS = {"base": "op_base", "copy": "op_copy", "grad": "grad"}
_MODS = {}
for _k, _p in _PKGS.items():
    if not (HERE / (_p + _NZ)).exists():
        continue
    _MODS[_k] = (
        importlib.import_module(_BASE + _p + _NZ + ".conv2d_3x3"),
        importlib.import_module(_BASE + _p + _NZ + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(_MODS[_k][1].KERNEL_DIR).parent.name == _p + _NZ, _MODS[_k][1].KERNEL_DIR
op, pd = _MODS["copy"]

VARIANTS = {
    "baseline": ("base", {}),
    "cand": ("copy", dict(S2_TAPS=True)),
    "cand_ub8": ("copy", dict(S2_TAPS=True, S2_FILL_BATCH_ROWS=8)),
    "cand_ub16": ("copy", dict(S2_TAPS=True, S2_FILL_BATCH_ROWS=16)),
    "cand_ub2": ("copy", dict(S2_TAPS=True, S2_FILL_BATCH_ROWS=2)),
    "off": ("copy", dict(S2_TAPS=False)),
    "grad": ("grad", {}),
}


def apply_variant(variant, monkeypatch):
    global op, pd
    pkg, knobs = VARIANTS[variant]
    op, pd = _MODS[pkg]
    for k, v in knobs.items():
        assert hasattr(pd, k), k
        monkeypatch.setattr(pd, k, v)
    for name in dir(pd):
        m = getattr(pd, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    import re

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            name = c["extras"]["layer"].replace(" @ LoFi [perf]", "")
            key = name.split(" ")[0].replace(".", "_")
            if (
                key in ("semantic", "instance", "center_head_1", "aspp_convs_1", "aspp_convs_2", "aspp_convs_3")
                or "{" in key
            ):
                key = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
            out[key] = c
    return out


FOCUS = _focus_cases()
# Stride-2 shapes of the perf / unit tests and the golden grid (HiFi4 unless noted) on the paths S2_TAPS reaches.
# (((H, W, C_in), C_out, padding, stride), memory layout, shard_shape, grid (x, y) or num_cores, dtype, fidelity)
EXTRA = {
    "res3_0_hifi4": (((128, 256, 128), 128, 1, 2), "height", [384, 128], 86, ttnn.bfloat8_b, "HiFi4"),
    "res4_0_hifi4": (((64, 128, 256), 256, 1, 2), "block", [1024, 32], (8, 8), ttnn.bfloat8_b, "HiFi4"),
    # golden res3.0-like downsample (32x64x128 -> 16x32x128) on the layouts the golden grid runs
    "g_s2_height_bf8": (((32, 64, 128), 128, 1, 2), "height", [256, 128], 8, ttnn.bfloat8_b, "HiFi4"),
    "g_s2_height_bf16": (((32, 64, 128), 128, 1, 2), "height", [256, 128], 8, ttnn.bfloat16, "HiFi4"),
    "g_s2_block_bf8": (((32, 64, 128), 128, 1, 2), "block", [256, 32], (4, 8), ttnn.bfloat8_b, "HiFi4"),
    "g_s2_dram_bf8": (((32, 64, 128), 128, 1, 2), "dram", None, None, ttnn.bfloat8_b, "HiFi4"),
    "g_s2_l1_bf16": (((32, 64, 128), 128, 1, 2), "l1", None, None, ttnn.bfloat16, "HiFi4"),
    # res3.0 at LoFi on a non-pinned grid (more rows per core)
    "res3_0_32c": (((128, 256, 128), 128, 1, 2), "height", [1024, 128], 32, ttnn.bfloat8_b, "LoFi"),
    # several M-blocks per core: res3.0 on 8 cores (32 output tile-rows each), and res4.0 / res3.0 DRAM interleaved
    "res3_0_16c": (((128, 256, 128), 128, 1, 2), "height", [2048, 128], 16, ttnn.bfloat8_b, "LoFi"),
    # res3_0_32c forced to 2 / 4 M-blocks per core (_footprint fails bm > FORCE_BM[case] and bk below a kernel row)
    "res3_0_32c_mb2": (((128, 256, 128), 128, 1, 2), "height", [1024, 128], 32, ttnn.bfloat8_b, "LoFi"),
    "res3_0_32c_mb4": (((128, 256, 128), 128, 1, 2), "height", [1024, 128], 32, ttnn.bfloat16, "HiFi4"),
    "g_s2_dram_mb": (((32, 64, 128), 128, 1, 2), "dram", None, None, ttnn.bfloat8_b, "HiFi4"),
    "res3_0_22c": (((128, 256, 128), 128, 1, 2), "height", [1504, 128], 22, ttnn.bfloat8_b, "HiFi4"),
    "res4_0_dram": (((64, 128, 256), 256, 1, 2), "dram", None, None, ttnn.bfloat8_b, "LoFi"),
    "res3_0_dram_bf16": (((128, 256, 128), 128, 1, 2), "dram", None, None, ttnn.bfloat16, "LoFi"),
}
DOMAIN = ["res3_0_conv2", "res4_0_conv2", "res3_0_hifi4", "res4_0_hifi4"] + [
    "g_s2_height_bf8",
    "g_s2_height_bf16",
    "g_s2_block_bf8",
    "g_s2_dram_bf8",
    "g_s2_l1_bf16",
    "res3_0_32c",
]


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("S2_CASES", "res3_0_conv2").split(",") if c]:
        out += DOMAIN if c == "domain" else (list(FOCUS) if c == "focus" else [c])
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("S2_VARIANTS", "baseline,cand").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("S2_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("S2_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("S2_NOCHECK") == "1"


def _run_focus(case, device):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    memory_config = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip(f"{extras['layer']}: pinned shard geometry does not fit this device's grid")
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
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    tt_output = op.conv2d_3x3(
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
    )
    pcc = (extras.get("tolerance") or (0.99, None))[0]
    return tt_output, ref, pcc, memory_config


def _grid(num_cores, grid_x=11):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _run_extra(name, device):
    import torch

    torch.manual_seed(0)
    ((H, W, c_in), c_out, padding, stride), ml, shard, grid, dtype, fidelity = EXTRA[name]
    if ml == "height":
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_grid(grid), shard, ttnn.ShardOrientation.ROW_MAJOR),
        )
    elif ml == "block":
        gx, gy = grid
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
                shard,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    elif ml == "dram":
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        memory_config = ttnn.L1_MEMORY_CONFIG
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=1,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=1,
        activation="relu",
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    x = ttnn.to_torch(tt_input).float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), stride=stride, padding=padding)
    ref = torch.relu(ref).permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return tt_output, ref, 0.99, memory_config


FORCE_BM = {"res3_0_32c_mb2": 4, "res3_0_32c_mb4": 2}


def _force_bm(monkeypatch, case_id):
    if case_id not in FORCE_BM:
        return
    real = pd._footprint
    cap = FORCE_BM[case_id]

    def capped(g, core_m_tiles, bm, bk, *a, **k):
        fit = real(g, core_m_tiles, bm, bk, *a, **k)
        if bm > cap or bk < g.tensor_kkh_tiles:
            fit = dict(fit, footprint_bytes=1 << 40)
        return fit

    monkeypatch.setattr(pd, "_footprint", capped)


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            nkb=b.num_k_blocks,
            nslice=bool(b.nslice),
            k_cols=g.k_cols,
            slab_rows=b.slab_tile_rows,
            ub=b.untilize_batch_rows,
            mblocks=b.max_m_blocks_per_core,
            ring=bool(b.ring),
            wdepth=b.weight_depth,
            core_m=b.core_m_tiles,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_s2(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    _force_bm(monkeypatch, case_id)
    if case_id in EXTRA:
        tt_output, ref, pcc_threshold, in_mc = _run_extra(case_id, device)
    else:
        tt_output, ref, pcc_threshold, in_mc = _run_focus(FOCUS[case_id], device)
    info["dbg"] = dict(getattr(pd, "DEBUG_INFO", {}))
    out_mc = tt_output.memory_config()
    if in_mc.shard_spec is not None:
        assert out_mc.memory_layout == in_mc.memory_layout
        info["out_spec"] = [
            str(out_mc.shard_spec.grid),
            list(out_mc.shard_spec.shape),
            str(out_mc.shard_spec.orientation),
        ]
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    sha = hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()[:16]
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        mae=float((out - ref.float()).abs().mean()),
        max_err=float((out - ref.float()).abs().max()),
        out_sha=sha,
    )
    outs = HERE / "runs" / "outs"
    outs.mkdir(parents=True, exist_ok=True)
    torch.save(out.to(torch.bfloat16), outs / f"{case_id}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
