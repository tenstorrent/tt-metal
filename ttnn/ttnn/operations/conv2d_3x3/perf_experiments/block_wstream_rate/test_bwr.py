# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the BLOCK N-slice column weight stream's round rate (perf experiment, round 12, `block_wstream_rate`).

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 739a271172e, unmodified copy)
  <name>    op_copy/ with the knobs in VARIANTS (module constants of op_copy's descriptor / op module)
BWR_NZ=1 selects the zone-stripped twins (op_base_nz / op_copy_nz, mk_twins.sh); BWR_ABL=1 the scatter-stubbed twins
(op_base_abl / op_copy_abl: the ring hop's NoC writes are skipped, so the stream's own ceiling shows; PCC unchecked).

Env: BWR_CASES (focus ids or `block:<layer>`, comma list; `domain` = the sweep), BWR_VARIANTS (comma list, run in that
order per case), BWR_ORDER=alt (reverse the order on every other case), BWR_LOG (run-log name), BWR_NOCHECK=1. Each
dispatch appends a JSON record (PCC, mean / max abs error vs a fp32 torch conv, output hash, the output .pt) to
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.block_wstream_rate."
_SUFFIX = ("_abl" if os.environ.get("BWR_ABL") == "1" else "") + ("nz" if os.environ.get("BWR_NZ") == "1" else "")
_SUFFIX = "_nz" if _SUFFIX == "nz" else _SUFFIX
_PKGS = {"base": _BASE + "op_base" + _SUFFIX, "copy": _BASE + "op_copy" + _SUFFIX}
if (HERE / ("grad2" + _SUFFIX)).exists():
    _PKGS["grad"] = _BASE + "grad2" + _SUFFIX
_MODS = {}
for _k, _p in _PKGS.items():
    _MODS[_k] = (
        importlib.import_module(_p + ".conv2d_3x3"),
        importlib.import_module(_p + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(_MODS[_k][1].KERNEL_DIR).parent.name == _p.rsplit(".", 1)[1], _MODS[_k][1].KERNEL_DIR
op, pd = _MODS["copy"]

# op_copy knobs (descriptor module constants unless prefixed "op."), reset to these before every dispatch.
KNOB_DEFAULTS = dict(
    BWR_DIAG=False,
    BWR_FWD_EARLY_ACK=False,
    BWR_SLOT_FLAGS=False,
    BWR_PREFETCH_ALL=False,
    BWR_PAGE_RUNS=False,
    BWR_MCAST_NB=False,
    BWR_FWD_ROUND0=False,
    BWR_SPARE_ROUND0=False,
    BWR_FA_SKIP0=False,
)
VARIANTS = {
    "baseline": ("base", {}),
    "grad": ("grad", {}),
    # op_copy with every knob off: must behave as HEAD.
    "copy": ("copy", {}),
    # HEAD plus split zones on the sender / forwarder / receivers (diagnosis only).
    "diag": ("copy", dict(BWR_DIAG=True)),
    # BLOCK weights ND-sharded one round per shard; the sender reads a round as one run.
    "pr": ("copy", dict(BWR_PAGE_RUNS=True)),
    # The sender keeps every free slot's round in flight (rounds 0..depth-1 at kernel start).
    "pf": ("copy", dict(BWR_PREFETCH_ALL=True)),
    # The forwarder acks the column sender's next round before it forwards this one.
    "fa": ("copy", dict(BWR_FWD_EARLY_ACK=True)),
    "pr_fa": ("copy", dict(BWR_PAGE_RUNS=True, BWR_FWD_EARLY_ACK=True)),
    "pr_pf": ("copy", dict(BWR_PAGE_RUNS=True, BWR_PREFETCH_ALL=True)),
    "all3": ("copy", dict(BWR_PAGE_RUNS=True, BWR_PREFETCH_ALL=True, BWR_FWD_EARLY_ACK=True)),
    # pr_fa plus non-blocking multicasts (the sender and the forwarder return once the multicast is issued).
    "pr_fa_nb": ("copy", dict(BWR_PAGE_RUNS=True, BWR_FWD_EARLY_ACK=True, BWR_MCAST_NB=True)),
    # pr_fa_nb plus the forwarder reading round 0 from DRAM itself.
    "pr_fa_nb_r0": ("copy", dict(BWR_PAGE_RUNS=True, BWR_FWD_EARLY_ACK=True, BWR_MCAST_NB=True, BWR_FWD_ROUND0=True)),
    "fa_nb": ("copy", dict(BWR_FWD_EARLY_ACK=True, BWR_MCAST_NB=True)),
    # fa_nb, but round 1 is acked early only after round 0's forward (round 0's forward not contended by round 1).
    "fa_nb_s0": ("copy", dict(BWR_FWD_EARLY_ACK=True, BWR_MCAST_NB=True, BWR_FA_SKIP0=True)),
    "fa_nb_r0": ("copy", dict(BWR_FWD_EARLY_ACK=True, BWR_MCAST_NB=True, BWR_FWD_ROUND0=True)),
    "pr_fa_r0": ("copy", dict(BWR_PAGE_RUNS=True, BWR_FWD_EARLY_ACK=True, BWR_FWD_ROUND0=True)),
    "pr_fa_nb_r0_diag": (
        "copy",
        dict(BWR_PAGE_RUNS=True, BWR_FWD_EARLY_ACK=True, BWR_MCAST_NB=True, BWR_FWD_ROUND0=True, BWR_DIAG=True),
    ),
    # Slot-flag channels (cumulative acks, one data-ready flag per slot, non-blocking multicasts) with page runs.
    "pr_sf": ("copy", dict(BWR_PAGE_RUNS=True, BWR_SLOT_FLAGS=True)),
    "pr_pf_sf": ("copy", dict(BWR_PAGE_RUNS=True, BWR_PREFETCH_ALL=True, BWR_SLOT_FLAGS=True)),
    "pr_sf_diag": ("copy", dict(BWR_PAGE_RUNS=True, BWR_SLOT_FLAGS=True, BWR_DIAG=True)),
    "sf": ("copy", dict(BWR_SLOT_FLAGS=True)),
    # pr_sf plus the spare members reading their round 0 from DRAM themselves.
    "pr_sf_r0": ("copy", dict(BWR_PAGE_RUNS=True, BWR_SLOT_FLAGS=True, BWR_SPARE_ROUND0=True)),
    "pr_pf_sf_r0": (
        "copy",
        dict(BWR_PAGE_RUNS=True, BWR_PREFETCH_ALL=True, BWR_SLOT_FLAGS=True, BWR_SPARE_ROUND0=True),
    ),
    "pr_sf_r0_diag": ("copy", dict(BWR_PAGE_RUNS=True, BWR_SLOT_FLAGS=True, BWR_SPARE_ROUND0=True, BWR_DIAG=True)),
    "all3_diag": ("copy", dict(BWR_PAGE_RUNS=True, BWR_PREFETCH_ALL=True, BWR_FWD_EARLY_ACK=True, BWR_DIAG=True)),
}


def apply_variant(variant, monkeypatch):
    global op, pd
    pkg, knobs = VARIANTS[variant]
    op, pd = _MODS[pkg]
    if pkg == "copy":
        for k, v in {**KNOB_DEFAULTS, **knobs}.items():
            mod, name = (op, k[3:]) if k.startswith("op.") else (pd, k)
            assert hasattr(mod, name), k
            monkeypatch.setattr(mod, name, v)
    for mod in (op, pd):
        for name in dir(mod):
            m = getattr(mod, name)
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
FOCUS_ID = "res4_{1__5}_conv2"
DOMAIN = [
    "res4_{1__5}_conv2",
    "res5_0_conv2",
    "res5_1_conv2",
    "res5_2_conv2",
    "aspp_convs_1",
    "aspp_convs_2",
    "aspp_convs_3",
]
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride): the HiFi4
# BLOCK shapes of test_conv2d_3x3_block_sharded_perf, plus its LoFi res4.
BLOCK_SHAPES = {
    "res4_0_conv2": (((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    "res4_conv2": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    "res4_conv2_lofi": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "LoFi", 1),
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
    # Not Panoptic: 2 channel slices, C_out 32 (one owner), d 8.
    "plain_k2_d8": (((32, 64, 64), 32, 8, 8), [256, 32], (2, 8), True, "LoFi", 1),
    # Not Panoptic: 4 channel slices, C_out 128, LoFi (spare groups).
    "plain_k4": (((32, 64, 128), 128, 1, 1), [256, 32], (4, 8), True, "LoFi", 1),
    # Not Panoptic: C_out 96 (3-tile rounds over 8 channel slices), a ragged last slice, LoFi.
    "plain_k8_co96": (((32, 64, 256), 96, 1, 1), [256, 32], (8, 8), True, "LoFi", 1),
}


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("BWR_CASES", FOCUS_ID).split(",") if c]:
        if c == "domain":
            out += DOMAIN
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("BWR_VARIANTS", "baseline,copy").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("BWR_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("BWR_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("BWR_NOCHECK") == "1" or os.environ.get("BWR_ABL") == "1"


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
    return tt_output, ref, pcc, memory_config, tt_weight


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
    return tt_output, ref, 0.99, memory_config, tt_weight


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_bwr(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold, in_mc, tt_weight = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold, in_mc, tt_weight = _run_focus(FOCUS[case_id], device)
    info["weight_mc"] = str(tt_weight.memory_config().nd_shard_spec)[:160]
    out_mc = tt_output.memory_config()
    assert out_mc.memory_layout == in_mc.memory_layout
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
