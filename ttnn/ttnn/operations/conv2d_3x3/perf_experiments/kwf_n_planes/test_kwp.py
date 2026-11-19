# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: kw-fold N planes on the 86-core HEIGHT layers (perf experiment, round 12, `kwf_n_planes`).

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 739a271172e, unmodified copy)
  planes    op_copy/ with KWF_N_PLANES = 2 (C_out split into 2 planes of N / 2 columns on ~104 cores)
  noplanes  op_copy/ with KWF_N_PLANES = 1 (must behave as HEAD)
  grad      grad/ (HEAD + graduation.patch)
KWP_NZ=1 selects the zone-stripped twins (*_nz, mk_nz.sh) for the unzoned verdict numbers.

Env: KWP_CASES (comma list of case keys below; `focus4` = the four 86-core focus cases, `domain` = the sweep),
KWP_VARIANTS (comma list, run in that order per case), KWP_ORDER=alt (reverse the order on every other case), KWP_LOG,
KWP_NOCHECK=1. Each dispatch appends a JSON record (plan, PCC, mean / max abs error vs a fp32 torch conv, output hash)
to runs/<log>.jsonl and saves the output to runs/outs/, which report.py joins with the profiler CSV's GenericOp rows in
call order. The precision contract is the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.kwf_n_planes."
_NZ = "_nz" if os.environ.get("KWP_NZ") == "1" else ""
_PKGS = {"base": "op_base", "copy": "op_copy", "grad": "grad", "base_abl": "op_base_abl", "copy_abl": "op_copy_abl"}
_MODS = {}
for _k, _p in _PKGS.items():
    if not (HERE / (_p + _NZ)).exists() or (_NZ and _p.endswith("_abl")):
        continue
    _MODS[_k] = (
        importlib.import_module(_BASE + _p + _NZ + ".conv2d_3x3"),
        importlib.import_module(_BASE + _p + _NZ + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(_MODS[_k][1].KERNEL_DIR).parent.name == _p + _NZ, _MODS[_k][1].KERNEL_DIR
op, pd = _MODS["base"]

KNOB_DEFAULTS = dict(
    KWF_N_PLANES=2,
    KWF_PLANES_LAYOUT="cols",
    KWF_PLANES_SPLIT="grid",
    KWF_PLANES_MODE="split",
    KWF_PLANES_WEIGHT_DEPTH=0,
    KWF_PLANES_TABLE=str(HERE / "layout_annealed.json"),
)
VARIANTS = {
    "baseline": ("base", {}),
    "grad": ("grad", {}),
    # "split": per-plane kw-fold (C_out / 2) with per-plane weight servers, column-block planes.
    "planes": ("copy", dict(KWF_N_PLANES=2)),
    # "stream": every core one n-group of HEAD's kw-fold, one n-group-interleaved weight stream to all work cores.
    "stream": ("copy", dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="pairs")),
    "stream_table": ("copy", dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="table")),
    "stream_table_d4": (
        "copy",
        dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="table", KWF_PLANES_WEIGHT_DEPTH=4),
    ),
    "stream_table_d6": (
        "copy",
        dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="table", KWF_PLANES_WEIGHT_DEPTH=6),
    ),
    # split mode on the per-plane blocks layout (fill_model.py blocks: annealed M order + split per core)
    "planes_blocks": (
        "copy",
        dict(KWF_N_PLANES=2, KWF_PLANES_LAYOUT="table", KWF_PLANES_TABLE=str(HERE / "layout_blocks.json")),
    ),
    "stream_cols": ("copy", dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="cols")),
    "stream_rows": ("copy", dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="rows")),
    "planes_rows": ("copy", dict(KWF_N_PLANES=2, KWF_PLANES_LAYOUT="rows")),
    "planes_rsplit": ("copy", dict(KWF_N_PLANES=2, KWF_PLANES_SPLIT="rows")),
    "noplanes": ("copy", dict(KWF_N_PLANES=1)),
    # Ablation-hooked twins (CONV2D_3X3_ABLATE=..., KWP_NOCHECK=1).
    "baseline_abl": ("base_abl", {}),
    "planes_abl": ("copy_abl", dict(KWF_N_PLANES=2)),
    "stream_abl": ("copy_abl", dict(KWF_N_PLANES=2, KWF_PLANES_MODE="stream", KWF_PLANES_LAYOUT="pairs")),
}


def apply_variant(variant, monkeypatch):
    global op, pd
    pkg, knobs = VARIANTS[variant]
    op, pd = _MODS[pkg]
    if pkg.startswith("copy"):
        for k, v in {**KNOB_DEFAULTS, **knobs}.items():
            assert hasattr(pd, k), k
            monkeypatch.setattr(pd, k, v)
    for name in dir(pd):
        m = getattr(pd, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()


# Focus cases by layer-name prefix (feature_spec "[perf]" rows).
FOCUS_KEYS = {
    "res3x": "res3.{1,2,3}.conv2 + instance res3.fuse_conv.1",
    "semdec3fc0": "semantic decoder.res3.fuse_conv.0",
    "semdec3fc1": "semantic decoder.res3.fuse_conv.1",
    "instdec3fc0": "instance decoder.res3.fuse_conv.0",
    "res2": "res2.{0,1,2}.conv2",
    "center1": "center_head.1 + offset_head.1",
    "instres2fc1": "instance res2.fuse_conv.1",
    "semres2fc1": "semantic res2.fuse_conv.1",
    "semdec2fc0": "semantic decoder.res2.fuse_conv.0",
    "instdec2fc0": "instance decoder.res2.fuse_conv.0",
    "stem2": "stem.conv2",
    "stem3": "stem.conv3",
    "res3_0": "res3.0.conv2",
}


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for key, prefix in FOCUS_KEYS.items():
        hits = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"] and c["extras"]["layer"].startswith(prefix)]
        assert len(hits) == 1, (key, [c["extras"]["layer"] for c in hits])
        out[key] = hits[0]
    return out


FOCUS = _focus_cases()
# The HiFi4 HEIGHT shapes of test_conv2d_3x3_height_sharded_perf (bf8_b, bias, ReLU):
# ((H, W, C_in), C_out, padding, stride), shard_shape, num_cores.
HEIGHT_SHAPES = {
    "h:res3": (((64, 128, 128), 128, 1, 1), [96, 128], 86),
    "h:semres3fc0": (((64, 128, 320), 256, 1, 1), [96, 320], 86),
    "h:instres3fc0": (((64, 128, 320), 128, 1, 1), [96, 320], 86),
    "h:semres3fc1": (((64, 128, 256), 256, 1, 1), [96, 256], 86),
    "h:res2": (((128, 256, 64), 64, 1, 1), [320, 64], 103),
    "h:center1": (((128, 256, 128), 32, 1, 1), [320, 128], 103),
}
FOCUS4 = ["res3x", "semdec3fc0", "semdec3fc1", "instdec3fc0"]
DOMAIN = FOCUS4 + ["h:res3", "h:semres3fc0", "h:instres3fc0", "h:semres3fc1", "res2", "center1", "instres2fc1"]
# The server-fed HEIGHT layers (the per-slot ack fix's reach), focus + HiFi4 harness shapes.
SERVERS = [
    "res3x",
    "semdec3fc0",
    "semdec3fc1",
    "instdec3fc0",
    "res2",
    "center1",
    "instres2fc1",
    "semres2fc1",
    "semdec2fc0",
    "instdec2fc0",
    "res3_0",
    "h:res3",
    "h:semres3fc0",
    "h:instres3fc0",
    "h:semres3fc1",
    "h:res2",
    "h:center1",
]


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("KWP_CASES", "res3x").split(",") if c]:
        if c == "focus4":
            out += FOCUS4
        elif c == "domain":
            out += DOMAIN
        elif c == "servers":
            out += SERVERS
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("KWP_VARIANTS", "baseline,planes").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("KWP_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("KWP_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("KWP_NOCHECK") == "1"


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


def _row_major_cores(num_cores, grid_x):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _run_height(key, device):
    import torch

    torch.manual_seed(0)
    ((H, W, c_in), c_out, padding, stride), shard_shape, num_cores = HEIGHT_SHAPES[key]
    grid = device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y or grid.x < 11:
        pytest.skip(f"{key}: pinned 11-wide {num_cores}-core shard grid does not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_row_major_cores(num_cores, 11), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=1,
        input_dtype=ttnn.bfloat8_b,
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
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    tt_output = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        activation="relu",
    )
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.relu(
        torch.nn.functional.conv2d(x, torch_weight.float(), torch_bias.float(), stride=stride, padding=padding)
    )
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return tt_output, ref, 0.99, memory_config


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_kwp(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant)
    if case_id.startswith("h:"):
        tt_output, ref, pcc_threshold, in_mc = _run_height(case_id, device)
    else:
        tt_output, ref, pcc_threshold, in_mc = _run_focus(FOCUS[case_id], device)
    info["plan"] = dict(getattr(pd, "DEBUG_INFO", {}))
    out_mc = tt_output.memory_config()
    # The output keeps the input's shard grid and orientation (its shard shape is HEAD's: compared bit-exactly, with the
    # recorded out_spec, against the baseline variant).
    assert out_mc.memory_layout == in_mc.memory_layout
    assert str(out_mc.shard_spec.grid) == str(in_mc.shard_spec.grid)
    assert out_mc.shard_spec.orientation == in_mc.shard_spec.orientation
    info["out_spec"] = [str(out_mc.shard_spec.grid), list(out_mc.shard_spec.shape), str(out_mc.shard_spec.orientation)]
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
    torch.save(out.to(torch.bfloat16), outs / f"{case_id.replace(':', '_')}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
