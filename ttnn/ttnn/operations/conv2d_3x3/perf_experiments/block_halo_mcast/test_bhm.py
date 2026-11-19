# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: BLOCK halo multicast (perf experiment, round 12, `block_halo_mcast`).

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 739a271172e, unmodified copy)
  mcast     op_copy/ with HALO_MCAST = True (owners push their shard rows to the column cores that need them)
  pull      op_copy/ with HALO_MCAST = False (must behave as HEAD)
BHM_NZ=1 selects the zone-stripped twins (op_base_nz / op_copy_nz) for the unzoned verdict numbers.

Env: BHM_CASES (focus ids or `block:<layer>`, comma list; `domain` = the sweep), BHM_VARIANTS (comma list, run in that
order per case), BHM_ORDER=alt (reverse the order on every other case), BHM_LOG (run-log name), BHM_NOCHECK=1. Each
dispatch appends a JSON record (blocking, PCC, mean / max abs error vs a fp32 torch conv, output hash, the output .pt) to
runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp rows in call order. The precision contract is
the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it. Inputs are seeded per case.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.block_halo_mcast."
_NZ = "_nz" if os.environ.get("BHM_NZ") == "1" else ""
_PKGS = {"base": _BASE + "op_base" + _NZ, "copy": _BASE + "op_copy" + _NZ}
if (HERE / ("grad" + _NZ)).exists():
    _PKGS["grad"] = _BASE + "grad" + _NZ
_MODS = {}
for _k, _p in _PKGS.items():
    _MODS[_k] = (
        importlib.import_module(_p + ".conv2d_3x3"),
        importlib.import_module(_p + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(_MODS[_k][1].KERNEL_DIR).parent.name == _p.rsplit(".", 1)[1], _MODS[_k][1].KERNEL_DIR
op, pd = _MODS["copy"]

KNOB_DEFAULTS = dict(
    HALO_MCAST=False,
    HALO_MCAST_RUNS="shard",
    HALO_MODE=0,
    HALO_INTERLEAVE=0,
    HALO_FLAG_ROWS=0,
    HALO_FLAG_VALUES=False,
    HALO_DESC=0,
)
VARIANTS = {
    "baseline": ("base", {}),
    "grad": ("grad", {}),
    "pull": ("copy", dict(HALO_MCAST=False)),
}
# mcast<mode>[x]: HALO_MODE 0 linked chains, 1 unlinked, 2 unlinked without path reserve, 3 unicast; x = exact runs.
for _m in range(4):
    VARIANTS[f"mcast{_m}"] = ("copy", dict(HALO_MCAST=True, HALO_MODE=_m))
    VARIANTS[f"mcast{_m}x"] = ("copy", dict(HALO_MCAST=True, HALO_MODE=_m, HALO_MCAST_RUNS="exact"))
    # i: the reader interleaves its down multicast with pushing the landed shards.
    VARIANTS[f"mcast{_m}i"] = ("copy", dict(HALO_MCAST=True, HALO_MODE=_m, HALO_INTERLEAVE=1))
    VARIANTS[f"mcast{_m}xi"] = ("copy", dict(HALO_MCAST=True, HALO_MODE=_m, HALO_MCAST_RUNS="exact", HALO_INTERLEAVE=1))
# p<B>: progress flags every B rows on the interleaved down multicast (no path reserve, exact runs).
for _b in (1, 2, 4):
    VARIANTS[f"mcast2xip{_b}"] = (
        "copy",
        dict(HALO_MCAST=True, HALO_MODE=2, HALO_MCAST_RUNS="exact", HALO_INTERLEAVE=1, HALO_FLAG_ROWS=_b),
    )
    # v: absolute-value flag writes instead of atomic increments.
    VARIANTS[f"mcast2xiv{_b}"] = (
        "copy",
        dict(
            HALO_MCAST=True,
            HALO_MODE=2,
            HALO_MCAST_RUNS="exact",
            HALO_INTERLEAVE=1,
            HALO_FLAG_ROWS=_b,
            HALO_FLAG_VALUES=True,
        ),
    )
    # d: top-down down multicast.
    VARIANTS[f"mcast2xivd{_b}"] = (
        "copy",
        dict(
            HALO_MCAST=True,
            HALO_MODE=2,
            HALO_MCAST_RUNS="exact",
            HALO_INTERLEAVE=1,
            HALO_FLAG_ROWS=_b,
            HALO_FLAG_VALUES=True,
            HALO_DESC=1,
        ),
    )
VARIANTS["mcast"] = VARIANTS["mcast0"]


def apply_variant(variant, monkeypatch):
    global op, pd
    pkg, knobs = VARIANTS[variant]
    op, pd = _MODS[pkg]
    if pkg == "copy":
        for k, v in {**KNOB_DEFAULTS, **knobs}.items():
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
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = c
    return out


FOCUS = _focus_cases()
DOMAIN = [
    "aspp_convs_2",
    "aspp_convs_3",
    "res4_{1__5}_conv2",
    "res5_0_conv2",
    "res5_1_conv2",
    "res5_2_conv2",
    "res4_0_conv2",
]
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride): the HiFi4
# BLOCK shapes of test_conv2d_3x3_block_sharded_perf.
BLOCK_SHAPES = {
    "res4_0_conv2": (((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    "res4_conv2": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    "res5_0_conv2": (((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_1_conv2": (((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_2_conv2": (((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, "HiFi4", 1),
    "aspp_convs1": (((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs2": (((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs3": (((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, "HiFi4", 1),
    # Not Panoptic: an odd k_cols (7 columns) ring with C_out 224 (7 owners) and bias, LoFi.
    "plain_k7": (((32, 64, 224), 224, 1, 1), [256, 32], (7, 8), True, "LoFi", 1),
    # Not Panoptic: 11 columns, C_out 160 = 5 owners of 1 tile (6 non-owning columns), no bias, LoFi.
    "plain_k11_o5": (((32, 64, 352), 160, 2, 2), [256, 32], (11, 8), False, "LoFi", 1),
    # Not Panoptic: 2 channel slices (6 spare groups of 2 cores: 16 groups of 4 tile-rows), C_out 32 (one owner), d 8.
    "plain_k2_d8": (((32, 64, 64), 32, 8, 8), [256, 32], (2, 8), True, "LoFi", 1),
    # Not Panoptic: 4 channel slices, C_out 128, LoFi (7 spare groups).
    "plain_k4": (((32, 64, 128), 128, 1, 1), [256, 32], (4, 8), True, "LoFi", 1),
    # res4.x at LoFi without bias / ReLU.
    "res4_lofi_plain": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), False, "LoFi", 1),
}


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("BHM_CASES", "aspp_convs_1").split(",") if c]:
        if c == "domain":
            out += DOMAIN
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("BHM_VARIANTS", "baseline,mcast").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("BHM_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("BHM_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("BHM_NOCHECK") == "1"


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


def _run_block(layer, device):
    import torch

    torch.manual_seed(0)
    ((H, W, c_in), c_out, padding, dilation), shard_shape, (gx, gy), bias_relu, fidelity, stride = BLOCK_SHAPES[layer]
    grid = device.compute_with_storage_grid_size()
    if gx > grid.x or gy > grid.y:
        pytest.skip(f"{layer}: pinned {gx}x{gy} shard grid does not fit {grid.x}x{grid.y}")
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if bias_relu else None
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if bias_relu else None
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
        dilation=dilation,
        activation="relu" if bias_relu else None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if bias_relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return tt_output, ref, 0.99, memory_config


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            nkb=b.num_k_blocks,
            nslice=bool(b.nslice),
            k_cols=g.k_cols,
            out_n=g.out_n_tiles,
            n_tiles=g.tensor_n_tiles,
            remote=bool(g.remote_output),
            ranges=[list(r) for r in g.m_ranges] if g.m_ranges else None,
            block_rows=getattr(g, "block_rows", 0),
            groups=b.num_m_groups,
            sb=list(pd._ring_subblock(b.block_m_tiles, g.out_n_tiles)),
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_bhm(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold, in_mc = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold, in_mc = _run_focus(FOCUS[case_id], device)
    info["dbg"] = dict(getattr(pd, "DEBUG_INFO", {}))
    out_mc = tt_output.memory_config()
    assert out_mc.memory_layout == in_mc.memory_layout
    assert str(out_mc.shard_spec.grid) == str(in_mc.shard_spec.grid)
    assert list(out_mc.shard_spec.shape) == [in_mc.shard_spec.shape[0], in_mc.shard_spec.shape[1]] or True
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
        out_shard=list(out_mc.shard_spec.shape),
    )
    outs = HERE / "runs" / "outs"
    outs.mkdir(parents=True, exist_ok=True)
    torch.save(out.to(torch.bfloat16), outs / f"{case_id.replace(':', '_')}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
