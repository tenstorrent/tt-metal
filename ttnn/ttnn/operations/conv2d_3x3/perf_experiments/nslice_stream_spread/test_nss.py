# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the BLOCK N-slice column weight stream spread over the column's members (perf experiment, round 14,
`nslice_stream_spread`).

Variants (one dispatch each, fresh program cache):
  baseline  op_base/ (HEAD 1a04504f992, unmodified copy)
  <name>    op_copy/ with the knobs in VARIANTS (module constants of op_copy's descriptor)
NSS_E_NZ=1 selects the zone-stripped twins (op_base_nz / op_copy_nz, mk_twins.py).

Env: NSS_E_CASES (focus ids or `block:<layer>`, comma list; `domain` = the sweep, `block` = the BLOCK perf shapes),
NSS_E_VARIANTS (comma list, run in that order per case), NSS_E_ORDER=alt (reverse the order on every other case),
NSS_E_LOG (run-log name), NSS_E_NOCHECK=1. Each dispatch appends a JSON record (PCC, mean / max abs error vs a fp32
torch conv, output hash, the output .pt) to runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp
rows in call order. The precision contract is the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it.
Inputs are seeded per case.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_stream_spread."
_SUFFIX = ("_abl" if os.environ.get("NSS_E_ABL") == "1" else "") + ("nz" if os.environ.get("NSS_E_NZ") == "1" else "")
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
    NSS_DIAG=False,
    NSS_SPREAD=False,
    NSS_READERS="spread",
    NSS_LOOKAHEAD=8,
    NSS_SKIP_RELAY=True,
    NSS_FREE=None,
    NSS_FASTADDR=False,
    NSS_RELAY="sender",
    NSS_ORDERED=0,
    NSLICE_WEIGHT_DEPTH=4,
)
VARIANTS = {
    "baseline": ("base", {}),
    # op_copy with every knob off: must behave as HEAD.
    "copy": ("copy", {}),
    # HEAD + graduation.patch (grad/).
    "grad": ("grad", {}),
    # HEAD plus split zones on the column stream (diagnosis only).
    "diag": ("copy", dict(NSS_DIAG=True)),
    # Spread stream: every round in its own slot, read and multicast raw by its own column member.
    "spread": ("copy", dict(NSS_SPREAD=True)),
    # Same, every round from the round-0 reader (one source per column, no handshake).
    "one": ("copy", dict(NSS_SPREAD=True, NSS_READERS="one")),
    # Spread readers, reads issued once round r - L landed on the reader.
    "spread_l1": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=1)),
    "spread_l2": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=2)),
    "spread_l3": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=3)),
    "spread_l4": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=4)),
    "one_l2": ("copy", dict(NSS_SPREAD=True, NSS_READERS="one", NSS_LOOKAHEAD=2)),
    # Only round 0 at walk setup; round r < L waits for round 0, r >= L for round r - L.
    "g0_l2": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=2, NSS_FREE=1)),
    "g0_l3": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=3, NSS_FREE=1)),
    "g0_l4": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=4, NSS_FREE=1)),
    # Per-round reader rows from the round-0 NoC1 link-sharing model.
    "m_l2": ("copy", dict(NSS_SPREAD=True, NSS_READERS="model", NSS_LOOKAHEAD=2)),
    "m_g0_l2": ("copy", dict(NSS_SPREAD=True, NSS_READERS="model", NSS_LOOKAHEAD=2, NSS_FREE=1)),
    "m_g0_l3": ("copy", dict(NSS_SPREAD=True, NSS_READERS="model", NSS_LOOKAHEAD=3, NSS_FREE=1)),
    # Rounds 0 and 1 from the round-0 reader at walk setup, round r >= 2 from its own member once r - L landed there.
    "pair_l2": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=2, NSS_FREE=2)),
    "pair_l3": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=3, NSS_FREE=2)),
    "pair_l4": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=4, NSS_FREE=2)),
    "pair_l2_diag": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_DIAG=True)),
    # Round 1 (and 2) only once round 0 landed on the reader.
    "pair_g0_l2": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=2, NSS_FREE=1)),
    "pair_l2_fa": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_FASTADDR=True)),
    "pair_l3_fa": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=3, NSS_FREE=2, NSS_FASTADDR=True)),
    # The baseline's roles (round 0 from the round-0 reader, the rest from the column sender; relays: round 0 the
    # column sender, the rest the forwarder) without any handshake: every round in its own slot.
    "base_l2": ("copy", dict(NSS_SPREAD=True, NSS_READERS="base", NSS_RELAY="base", NSS_LOOKAHEAD=2, NSS_FREE=2)),
    "base_l3": ("copy", dict(NSS_SPREAD=True, NSS_READERS="base", NSS_RELAY="base", NSS_LOOKAHEAD=3, NSS_FREE=2)),
    "base_l2_fa": (
        "copy",
        dict(NSS_SPREAD=True, NSS_READERS="base", NSS_RELAY="base", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_FASTADDR=True),
    ),
    "pair_l2_rb": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_RELAY="base", NSS_LOOKAHEAD=2, NSS_FREE=2)),
    # Ordered column multicasts (round r once r - 1 landed on its reader), fast addresses.
    "pair_o": (
        "copy",
        dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_FASTADDR=True, NSS_ORDERED=32),
    ),
    "pair_o3": (
        "copy",
        dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=3, NSS_FREE=2, NSS_FASTADDR=True, NSS_ORDERED=32),
    ),
    "base_o": (
        "copy",
        dict(
            NSS_SPREAD=True,
            NSS_READERS="base",
            NSS_RELAY="base",
            NSS_LOOKAHEAD=2,
            NSS_FREE=2,
            NSS_FASTADDR=True,
            NSS_ORDERED=32,
        ),
    ),
    "spread_o": (
        "copy",
        dict(NSS_SPREAD=True, NSS_LOOKAHEAD=2, NSS_FREE=1, NSS_FASTADDR=True, NSS_ORDERED=32),
    ),
    # One round in flight per column: round r read once r - 1 landed on its reader.
    "pair_l1": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pair", NSS_LOOKAHEAD=1, NSS_FREE=1, NSS_FASTADDR=True)),
    "spread_l1f": ("copy", dict(NSS_SPREAD=True, NSS_LOOKAHEAD=1, NSS_FREE=1, NSS_FASTADDR=True)),
    # Round 0 from the round-0 reader, round 1 from the column sender (the baseline's start-up readers), the rest
    # spread over the other rows.
    "pairb": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pairb", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_FASTADDR=True)),
    "pairb_l1": ("copy", dict(NSS_SPREAD=True, NSS_READERS="pairb", NSS_LOOKAHEAD=1, NSS_FREE=1, NSS_FASTADDR=True)),
    "pairb_o": (
        "copy",
        dict(NSS_SPREAD=True, NSS_READERS="pairb", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_FASTADDR=True, NSS_ORDERED=32),
    ),
    "pairb_o1": (
        "copy",
        dict(NSS_SPREAD=True, NSS_READERS="pairb", NSS_LOOKAHEAD=2, NSS_FREE=2, NSS_FASTADDR=True, NSS_ORDERED=1),
    ),
    # The baseline's stream with an 8-deep weight CB (the candidate's L1 footprint, no protocol change).
    "depth8": ("copy", dict(NSLICE_WEIGHT_DEPTH=8)),
    # Spread readers, the relay row reads too.
    "spread_rr": ("copy", dict(NSS_SPREAD=True, NSS_SKIP_RELAY=False)),
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
            # _round0_reader_rows is a pure function of its arguments (a search that is slow under tracy's Python
            # profiler): kept across dispatches.
            if callable(m) and hasattr(m, "cache_clear") and name != "_round0_reader_rows":
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
    for c in [c for c in os.environ.get("NSS_E_CASES", FOCUS_ID).split(",") if c]:
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
SEL_VARIANTS = [v for v in os.environ.get("NSS_E_VARIANTS", "baseline,copy").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("NSS_E_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("NSS_E_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("NSS_E_NOCHECK") == "1" or os.environ.get("NSS_E_ABL") == "1"


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
def test_nss(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold, in_mc, tt_weight = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold, in_mc, tt_weight = _run_focus(FOCUS[case_id], device)
    info["weight_mc"] = str(tt_weight.memory_config().nd_shard_spec)[:160]
    info["nss"] = getattr(pd, "NSS_LAST", "-")
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
    outs = HERE / "runs" / ("outs_abl" if os.environ.get("NSS_E_ABL") == "1" else "outs")
    outs.mkdir(parents=True, exist_ok=True)
    torch.save(out.to(torch.bfloat16), outs / f"{case_id.replace(':', '_')}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
