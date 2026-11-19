# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the BLOCK N-slice column weight stream's round 0 (perf experiment, round 13, `block_round0`).

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 4f2f03f6d20, unmodified copy)
  <name>    op_copy/ with the knobs in VARIANTS (module constants of op_copy's descriptor / op module)
BR0_NZ=1 selects the zone-stripped twins (op_base_nz / op_copy_nz, mk_twins.sh); BR0_ABL=1 the scatter-stubbed twins
(op_base_abl / op_copy_abl, mk_abl.py: CONV2D_3X3_ABLATE stubs, default `fill`; PCC unchecked; with BR0_NZ=1 the
zone-stripped _ablnz twins).

Env: BR0_CASES (focus ids or `block:<layer>`, comma list; `domain` = the sweep), BR0_VARIANTS (comma list, run in that
order per case), BR0_ORDER=alt (reverse the order on every other case), BR0_LOG (run-log name), BR0_NOCHECK=1. Each
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.block_round0."
_SUFFIX = ("_abl" if os.environ.get("BR0_ABL") == "1" else "") + ("nz" if os.environ.get("BR0_NZ") == "1" else "")
_SUFFIX = {"nz": "_nz", "_ablnz": "_ablnz"}.get(_SUFFIX, _SUFFIX)
_PKGS = {"base": _BASE + "op_base" + _SUFFIX, "copy": _BASE + "op_copy" + _SUFFIX}
if (HERE / ("grad" + _SUFFIX)).exists():
    _PKGS["grad"] = _BASE + "grad" + _SUFFIX
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
    BR0_DIAG=False,
    BR0_NO_PF0=False,
    BR0_LATE_BIAS=False,
    BR0_EARLY=False,
    BR0_DUAL=False,
    BR0_SENDER_SHIFT=0,
    BR0_Q=False,
    BR0_Q_ROWS=None,
    BR0_NOFLUSH0=False,
)
VARIANTS = {
    "baseline": ("base", {}),
    # op_copy with every knob off: must behave as HEAD.
    "copy": ("copy", {}),
    # HEAD + graduation.patch (grad/).
    "grad": ("grad", {}),
    # HEAD plus split zones on the column sender (diagnosis only).
    "diag": ("copy", dict(BR0_DIAG=True)),
    # The column sender issues round 1's read only once round 0's landed (no prefetch beside round 0).
    "nopf0": ("copy", dict(BR0_NO_PF0=True)),
    # The N-slice async bias is read once the core's round 0 landed (not at kernel start).
    "lb": ("copy", dict(BR0_LATE_BIAS=True)),
    "nopf0_lb": ("copy", dict(BR0_NO_PF0=True, BR0_LATE_BIAS=True)),
    "nopf0_diag": ("copy", dict(BR0_NO_PF0=True, BR0_DIAG=True)),
    "lb_diag": ("copy", dict(BR0_LATE_BIAS=True, BR0_DIAG=True)),
    "nopf0_lb_diag": ("copy", dict(BR0_NO_PF0=True, BR0_LATE_BIAS=True, BR0_DIAG=True)),
    # The column sender issues round 0's read before anything else of the N-slice walk.
    "early": ("copy", dict(BR0_EARLY=True)),
    "lb_early": ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True)),
    "lb_early_diag": ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DIAG=True)),
    "nopf0_lb_early": ("copy", dict(BR0_NO_PF0=True, BR0_LATE_BIAS=True, BR0_EARLY=True)),
    "nopf0_lb_early_diag": ("copy", dict(BR0_NO_PF0=True, BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DIAG=True)),
    # The column sender multicasts round 0 to its slice's spare members too (the forwarder skips round 0).
    "dual": ("copy", dict(BR0_DUAL=True)),
    "dual_diag": ("copy", dict(BR0_DUAL=True, BR0_DIAG=True)),
    "lb_early_dual": ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DUAL=True)),
    "lb_early_dual_diag": ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DUAL=True, BR0_DIAG=True)),
}


# Column-sender row sweep (the DRAM read's path depends on where the sender sits).
for _s in range(10):
    VARIANTS[f"led_s{_s}_diag"] = (
        "copy",
        dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DUAL=True, BR0_DIAG=True, BR0_SENDER_SHIFT=_s),
    )
    VARIANTS[f"led_s{_s}"] = ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DUAL=True, BR0_SENDER_SHIFT=_s))


# Round 0 from a per-column round-0 reader Q (raw, no handshake) to the column and the spare row (Q = the column sender,
# or the grid row index per column in BR0_Q_ROWS).
Q_ROWS = {
    "A": (0, 9, 7, 2, 5, 6, 8, 1),
    "B": (1, 9, 8, 2, 5, 7, 6, 0),
    "C": (2, 9, 7, 0, 5, 6, 8, 1),
    "E": (3, 9, 8, 2, 5, 7, 6, 0),
    "M": (5, 6, 8, 1, 4, 9, 7, 2),
}
VARIANTS["q"] = ("copy", dict(BR0_Q=True))
VARIANTS["nf"] = ("copy", dict(BR0_NOFLUSH0=True))
VARIANTS["led_nf"] = ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_DUAL=True, BR0_NOFLUSH0=True))
VARIANTS["le_nf"] = ("copy", dict(BR0_LATE_BIAS=True, BR0_EARLY=True, BR0_NOFLUSH0=True))
VARIANTS["q_lb"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True))
VARIANTS["q_lb_diag"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_DIAG=True))
VARIANTS["q_lb_nopf0"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_NO_PF0=True))
VARIANTS["qmod_lb"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_Q_ROWS="model"))
VARIANTS["qmod_lb_diag"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_Q_ROWS="model", BR0_DIAG=True))
VARIANTS["q_lb_nopf0_diag"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_NO_PF0=True, BR0_DIAG=True))
for _n, _r in Q_ROWS.items():
    VARIANTS[f"q{_n}_lb"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_Q_ROWS=_r))
    VARIANTS[f"q{_n}"] = ("copy", dict(BR0_Q=True, BR0_Q_ROWS=_r))
    VARIANTS[f"q{_n}_lb_diag"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_Q_ROWS=_r, BR0_DIAG=True))
    VARIANTS[f"q{_n}_lb_nopf0"] = ("copy", dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_Q_ROWS=_r, BR0_NO_PF0=True))
    VARIANTS[f"q{_n}_lb_nopf0_diag"] = (
        "copy",
        dict(BR0_Q=True, BR0_LATE_BIAS=True, BR0_Q_ROWS=_r, BR0_NO_PF0=True, BR0_DIAG=True),
    )


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
            k = c["extras"]["layer"].split(" ")[0].replace(".", "_")
            if k in out or k in ("semantic", "instance"):
                # The decoder layers share their first word: key them by the whole layer name.
                k = "_".join(c["extras"]["layer"].split(" @ ")[0].replace(".", "_").replace("+", "").split())
            out[k] = c
    return out


FOCUS = _focus_cases()
FOCUS_ID = "res4_{1__5}_conv2"
DOMAIN = [
    "res4_{1__5}_conv2",
    "res4_0_conv2",
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
    for c in [c for c in os.environ.get("BR0_CASES", FOCUS_ID).split(",") if c]:
        if c == "domain":
            out += DOMAIN
        elif c == "focus":
            out += list(FOCUS)
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("BR0_VARIANTS", "baseline,copy").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("BR0_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("BR0_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("BR0_NOCHECK") == "1" or os.environ.get("BR0_ABL") == "1"


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
    outs = HERE / "runs" / ("outs_abl" if os.environ.get("BR0_ABL") == "1" else "outs")
    outs.mkdir(parents=True, exist_ok=True)
    torch.save(out.to(torch.bfloat16), outs / f"{case_id.replace(':', '_')}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
