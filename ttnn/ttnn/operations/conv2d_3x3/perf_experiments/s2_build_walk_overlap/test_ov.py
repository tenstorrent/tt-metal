# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B (perf experiment, round 13, `s2_build_walk_overlap`): overlap the stride-2 DEST tap build with the matmul walk
instead of running them back to back on TRISC.

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 4f2f03f6d20, unmodified copy)
  cand*     op_copy/ with the descriptor knob S2_OVERLAP (0 = HEAD schedule)
  grad      grad/ (HEAD + graduation.patch), when present
OV_NZ=1 selects the zone-stripped twins (*_nz, mk_nz.sh) for the unzoned verdict numbers.

Env: OV_CASES (comma list of case ids; `focus` = all 22 [perf] cases, `domain`, `unit`), OV_VARIANTS (comma list, run
in that order per case), OV_ORDER=alt (reverse the order on every other case), OV_LOG (run-log name), OV_NOCHECK=1.
Each dispatch appends a JSON record (PCC vs a fp32 torch conv, output hash, the saved output) to runs/<log>.jsonl,
which report.py joins with the profiler CSV's GenericOp rows in call order. The precision contract is the case's: no
variant touches it. Inputs are seeded per case.
"""

import hashlib
import importlib
import json
import os
import re
import time
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.s2_build_walk_overlap."
_NZ = "_nz" if os.environ.get("OV_NZ") == "1" else ""
_PKGS = {"base": "op_base", "copy": "op_copy", "grad": "grad", "abl": "op_abl"}
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
    "off": ("copy", dict(S2_OVERLAP=0)),
    "cand": ("copy", dict(S2_OVERLAP=1)),
    "cand2": ("copy", dict(S2_OVERLAP=2)),
    "cand3": ("copy", dict(S2_OVERLAP=3)),
    "cand4": ("copy", dict(S2_OVERLAP=4)),
    "cand5": ("copy", dict(S2_OVERLAP=5)),
    "cand6": ("copy", dict(S2_OVERLAP=6)),
    "diag9": ("copy", dict(S2_OVERLAP=9)),
    "diag7": ("copy", dict(S2_OVERLAP=7)),
    "kbco": ("copy", dict(KB_CALLER_OWNED=True)),
    "kbco_w3": ("copy", dict(KB_CALLER_OWNED=True, S2_WDEPTH=3)),
    "off_w3": ("copy", dict(S2_OVERLAP=0, S2_WDEPTH=3)),
    "cand4_w3": ("copy", dict(S2_OVERLAP=4, S2_WDEPTH=3)),
    "cand6_w3": ("copy", dict(S2_OVERLAP=6, S2_WDEPTH=3)),
    "grad": ("grad", {}),
    # ablation (op_abl = op_base + ablation_r13/ablation_hooks.patch; output wrong by construction: OV_NOCHECK=1)
    "abl_full": ("abl", {}, ""),
    "abl_dm": ("abl", {}, "gather,fill,weights,scatter,untilize"),
    "abl_dm_tilize": ("abl", {}, "gather,fill,weights,scatter,untilize,tilize"),
    "abl_fill": ("abl", {}, "fill"),
    "abl_weights": ("abl", {}, "weights"),
    "abl_tilize": ("abl", {}, "tilize"),
}


def apply_variant(variant, monkeypatch):
    global op, pd
    pkg, knobs = VARIANTS[variant][:2]
    if len(VARIANTS[variant]) > 2:
        monkeypatch.setenv("CONV2D_3X3_ABLATE", VARIANTS[variant][2])
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
# (((H, W, C_in), C_out, padding, stride), memory layout, shard_shape, grid (x, y) or num_cores, dtype, fidelity, bm cap)
EXTRA = {
    # the perf file's HiFi4 twins of the two focus layers
    "res3_0_hifi4": (((128, 256, 128), 128, 1, 2), "height", [384, 128], 86, ttnn.bfloat8_b, "HiFi4", None),
    "res4_0_hifi4": (((64, 128, 256), 256, 1, 2), "block", [1024, 32], (8, 8), ttnn.bfloat8_b, "HiFi4", None),
    # the stride-2 unit file's cases (test_conv2d_3x3_stride2_taps.py)
    "u_height_rows": (((32, 64, 128), 128, 1, 2), "height", [256, 128], 8, ttnn.bfloat8_b, "HiFi4", None),
    "u_height_rows_bf16": (((32, 64, 128), 128, 1, 2), "height", [256, 128], 8, ttnn.bfloat16, "HiFi4", None),
    "u_height_odd_h": (((33, 64, 64), 64, 1, 2), "height", [352, 64], 6, ttnn.bfloat8_b, "HiFi4", None),
    "u_height_w128_two_rows": (((16, 128, 64), 96, 1, 2), "height", [192, 64], 11, ttnn.bfloat8_b, "HiFi4", None),
    "u_block_nslice": (((32, 128, 256), 256, 1, 2), "block", [1024, 32], (8, 4), ttnn.bfloat8_b, "HiFi4", None),
    "u_block_nslice_bf16": (((32, 64, 128), 128, 1, 2), "block", [512, 32], (4, 4), ttnn.bfloat16, "HiFi4", None),
    "u_dram": (((32, 64, 128), 128, 1, 2), "dram", None, None, ttnn.bfloat8_b, "HiFi4", None),
    "u_height_mblocks": (((64, 128, 128), 128, 1, 2), "height", [1024, 128], 8, ttnn.bfloat8_b, "HiFi4", 2),
    # res3.0 at LoFi on fewer cores (more tile-rows per core), and DRAM interleaved
    "res3_0_32c": (((128, 256, 128), 128, 1, 2), "height", [1024, 128], 32, ttnn.bfloat8_b, "LoFi", None),
    "res3_0_dram_bf16": (((128, 256, 128), 128, 1, 2), "dram", None, None, ttnn.bfloat16, "LoFi", None),
    "res4_0_dram": (((64, 128, 256), 256, 1, 2), "dram", None, None, ttnn.bfloat8_b, "LoFi", None),
    # stride-1 DRAM-interleaved shapes of the perf file (the K-blocked helper walk outside S2_TAPS)
    "i_64x128x128": (((64, 128, 128), 128, 1, 1), "dram", None, None, ttnn.bfloat8_b, "HiFi4", None),
    "i_32x64x128_bf16": (((32, 64, 128), 128, 1, 1), "dram", None, None, ttnn.bfloat16, "HiFi4", None),
    "i_8x16x2048_bf16": (((8, 16, 2048), 256, 1, 1), "dram", None, None, ttnn.bfloat16, "HiFi4", None),
    "i_16x16x64": (((16, 16, 64), 128, 1, 1), "dram", None, None, ttnn.bfloat8_b, "HiFi4", None),
    "i_128x256x64": (((128, 256, 64), 64, 1, 1), "dram", None, None, ttnn.bfloat8_b, "HiFi4", None),
}
DOMAIN = ["res3_0_hifi4", "res4_0_hifi4", "res3_0_32c", "res3_0_dram_bf16", "res4_0_dram"] + [
    "i_64x128x128",
    "i_32x64x128_bf16",
    "i_8x16x2048_bf16",
    "i_16x16x64",
    "i_128x256x64",
]
UNIT = [k for k in EXTRA if k.startswith("u_")]


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("OV_CASES", "res3_0_conv2").split(",") if c]:
        out += {"domain": DOMAIN, "focus": list(FOCUS), "unit": UNIT}.get(c, [c])
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("OV_VARIANTS", "baseline,cand").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("OV_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("OV_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("OV_NOCHECK") == "1"


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
    ((H, W, c_in), c_out, padding, stride), ml, shard, grid, dtype, fidelity, _ = EXTRA[name]
    if ml == "height":
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                _grid(grid, device.compute_with_storage_grid_size().x), shard, ttnn.ShardOrientation.ROW_MAJOR
            ),
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
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs)
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
    return tt_output, ref, (0.997 if dtype == ttnn.bfloat16 else 0.99), memory_config


def _force_bm(monkeypatch, case_id):
    cap = EXTRA[case_id][-1] if case_id in EXTRA else None
    if cap is None:
        return
    real = pd._footprint

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
            wdepth=b.weight_depth,
            core_m=b.core_m_tiles,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_ov(case_id, variant, device, monkeypatch):
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
    out_mc = tt_output.memory_config()
    if in_mc.shard_spec is not None:
        assert out_mc.memory_layout == in_mc.memory_layout
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
