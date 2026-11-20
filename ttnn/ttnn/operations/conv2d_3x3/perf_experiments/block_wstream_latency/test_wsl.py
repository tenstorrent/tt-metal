# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the BLOCK N-slice spread column weight stream's per-round latency to the spare-column members (perf
experiment, round 15, `block_wstream_latency`).

Variants (one dispatch each, fresh program cache), each (package, op_copy descriptor knobs):
  base     op_base/ (HEAD ec6f75b0d28 + ablation_r15 hooks, inactive unless a token is given)
  copy     op_copy/ with every knob at HEAD's value (must equal base)
  others   see VARIANTS
A `@tok+tok` suffix runs the variant with CONV2D_3X3_ABLATE=tok,tok (`_w` = weights, `_m` = matmul); ablated outputs
are wrong by construction and are not checked. WSL_NZ=1 selects the zone-stripped twins (op_base_nz / op_copy_nz,
mk_nz.sh).

Env: WSL_CASES (focus ids or `block:<layer>`, `;` list; `domain` = the sweep, `focus22`, `block`), WSL_VARIANTS (comma
list, run in that order per case), WSL_ORDER=alt, WSL_REPEAT=n, WSL_LOG, WSL_NOCHECK=1. Each dispatch appends a JSON
record (blocking, PCC, mean / max abs error vs a fp32 torch conv, output hash) to runs/<log>.jsonl and saves the output
to runs/outs/<case>-<variant>.pt; report.py joins them with the profiler CSV's GenericOp rows in call order. The
precision contract is the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it. Inputs are seeded per case.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.resolve()))
from op_copies import copy_modules  # noqa: E402

_NZ = "_nz" if os.environ.get("WSL_NZ") == "1" else ""
_MODS = {k: copy_modules(k + _NZ) for k in ("op_base", "op_copy", "grad") if (HERE / (k + _NZ)).exists()}
for _k, (_o, _p) in _MODS.items():
    assert Path(_p.KERNEL_DIR).parent.name == _k + _NZ, _p.KERNEL_DIR
op, pd = _MODS["op_copy"]

# name -> (package, op_copy descriptor knobs)
KNOB_DEFAULTS = dict(
    WSL_SPARE_MODE="relay",
    WSL_SELF_READERS="rot",
    WSL_SELF_SLICES=None,
    WSL_READERS="head",
    WSL_MODEL_WINDOW=2,
    WSL_MODEL_R1=True,
    NSLICE_SPREAD_LOOKAHEAD=2,
    WSL_RELAY_NOW=False,
    WSL_BIAS_ROUND=0,
    WSL_ORDERED=False,
    WSL_RELAY_FIRST=False,
    WSL_LATE_LOOKAHEAD=None,
    WSL_LATE_FROM=3,
    WSL_EARLY_ISSUE=False,
    WSL_DUP=False,
    WSL_MODEL_XLEG=True,
)
VARIANTS = {
    "base": ("op_base", {}),
    "copy": ("op_copy", {}),
    # The round's reading member multicasts it to the spare members too (no relay).
    "direct": ("op_copy", dict(WSL_SPARE_MODE="direct")),
    # The spare members read their slice's rounds themselves, in turn / all from the highest-x / lowest-x member.
    "self": ("op_copy", dict(WSL_SPARE_MODE="self")),
    "self_far": ("op_copy", dict(WSL_SPARE_MODE="self", WSL_SELF_READERS="far")),
    "self_near": ("op_copy", dict(WSL_SPARE_MODE="self", WSL_SELF_READERS="near")),
    # Column readers on the grid rows that host no relay only / round 1 off the sender's row.
    "free": ("op_copy", dict(WSL_READERS="free")),
    # Per-round column readers from a NoC1 path model (reads within the window share links), lookahead 2 / 3.
    "model": ("op_copy", dict(WSL_READERS="model")),
    "model_nr1": ("op_copy", dict(WSL_READERS="model", WSL_MODEL_R1=False)),
    "model_w3": ("op_copy", dict(WSL_READERS="model", WSL_MODEL_WINDOW=3)),
    "model_l3": ("op_copy", dict(WSL_READERS="model", WSL_MODEL_WINDOW=3, NSLICE_SPREAD_LOOKAHEAD=3)),
    "head_l3": ("op_copy", dict(NSLICE_SPREAD_LOOKAHEAD=3)),
    # The relay core relays the rounds it read itself right behind their column multicast.
    "now": ("op_copy", dict(WSL_RELAY_NOW=True)),
    "now_l3": ("op_copy", dict(WSL_RELAY_NOW=True, NSLICE_SPREAD_LOOKAHEAD=3)),
    "now_mnr1": ("op_copy", dict(WSL_RELAY_NOW=True, WSL_READERS="model", WSL_MODEL_R1=False)),
    "now_mnr1_l3": (
        "op_copy",
        dict(
            WSL_RELAY_NOW=True, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=3, NSLICE_SPREAD_LOOKAHEAD=3
        ),
    ),
    "now_sender1": ("op_copy", dict(WSL_RELAY_NOW=True, WSL_READERS="sender1")),
    # The async bias read issued at round 2's / round 7's push instead of round 0's.
    "bias2": ("op_copy", dict(WSL_BIAS_ROUND=2)),
    "bias7": ("op_copy", dict(WSL_BIAS_ROUND=7)),
    # Ordered column multicasts, lookahead 2 / 3, HEAD or model readers (round 1 on the sender).
    "ord_l2": ("op_copy", dict(WSL_ORDERED=True)),
    "ord_l3": ("op_copy", dict(WSL_ORDERED=True, NSLICE_SPREAD_LOOKAHEAD=3)),
    "ord_ml3": (
        "op_copy",
        dict(WSL_ORDERED=True, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=3, NSLICE_SPREAD_LOOKAHEAD=3),
    ),
    "ord_ml4": (
        "op_copy",
        dict(WSL_ORDERED=True, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=4, NSLICE_SPREAD_LOOKAHEAD=4),
    ),
    # Relay first (own rounds to the spare members before the column), adaptive lookahead (2, then 3 from round 3).
    "rf": ("op_copy", dict(WSL_RELAY_FIRST=True)),
    "ad": ("op_copy", dict(WSL_LATE_LOOKAHEAD=3)),
    "rf_ad": ("op_copy", dict(WSL_RELAY_FIRST=True, WSL_LATE_LOOKAHEAD=3)),
    "rf_ad_mdl": (
        "op_copy",
        dict(WSL_RELAY_FIRST=True, WSL_LATE_LOOKAHEAD=3, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=3),
    ),
    "rf_ad_mdlo": (
        "op_copy",
        dict(
            WSL_RELAY_FIRST=True,
            WSL_LATE_LOOKAHEAD=3,
            WSL_READERS="model",
            WSL_MODEL_R1=False,
            WSL_MODEL_WINDOW=3,
            WSL_ORDERED=True,
        ),
    ),
    "ad_mdl": (
        "op_copy",
        dict(WSL_LATE_LOOKAHEAD=3, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=3),
    ),
    "ad4": ("op_copy", dict(WSL_LATE_LOOKAHEAD=4, WSL_LATE_FROM=4)),
    # The stream's first reads before the writer's host fill.
    "early": ("op_copy", dict(WSL_EARLY_ISSUE=True)),
    "early_ad": ("op_copy", dict(WSL_EARLY_ISSUE=True, WSL_LATE_LOOKAHEAD=3)),
    # The relay core reads every round itself too.
    "dup": ("op_copy", dict(WSL_DUP=True)),
    "dup_ad": ("op_copy", dict(WSL_DUP=True, WSL_LATE_LOOKAHEAD=3)),
    "ord_ad": ("op_copy", dict(WSL_ORDERED=True, WSL_LATE_LOOKAHEAD=3)),
    "ord_ad_mdl": (
        "op_copy",
        dict(WSL_ORDERED=True, WSL_LATE_LOOKAHEAD=3, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=3),
    ),
    # ord_ml3 with the model's climbs only (no physical-x row leg).
    "ord_ml3c": (
        "op_copy",
        dict(
            WSL_ORDERED=True,
            WSL_READERS="model",
            WSL_MODEL_R1=False,
            WSL_MODEL_WINDOW=3,
            NSLICE_SPREAD_LOOKAHEAD=3,
            WSL_MODEL_XLEG=False,
        ),
    ),
    "mnr1_l3": (
        "op_copy",
        dict(WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=3, NSLICE_SPREAD_LOOKAHEAD=3),
    ),
    "now_mnr1_l4": (
        "op_copy",
        dict(
            WSL_RELAY_NOW=True, WSL_READERS="model", WSL_MODEL_R1=False, WSL_MODEL_WINDOW=4, NSLICE_SPREAD_LOOKAHEAD=4
        ),
    ),
    "now_m_l3": (
        "op_copy",
        dict(WSL_RELAY_NOW=True, WSL_READERS="model", WSL_MODEL_WINDOW=3, NSLICE_SPREAD_LOOKAHEAD=3),
    ),
    # The column sender (the relay) reads every round / every round after 0 / every other round.
    "sender": ("op_copy", dict(WSL_READERS="sender")),
    "sender1": ("op_copy", dict(WSL_READERS="sender1")),
    "sender_odd": ("op_copy", dict(WSL_READERS="sender_odd")),
    # The spare members fed by unicast: from the column sender (relay) / from each round's reading member (direct).
    "relay_uc": ("op_copy", dict(WSL_SPARE_MODE="relay_uc")),
    "direct_uc": ("op_copy", dict(WSL_SPARE_MODE="direct_uc")),
    "r1off_uc": ("op_copy", dict(WSL_SPARE_MODE="relay_uc", WSL_READERS="r1off")),
    "r1off_duc": ("op_copy", dict(WSL_SPARE_MODE="direct_uc", WSL_READERS="r1off")),
    "r1off": ("op_copy", dict(WSL_READERS="r1off")),
}
if (HERE / ("grad" + _NZ)).exists():
    VARIANTS["grad"] = ("grad", {})


def _variant(name):
    abl = ""
    if "@" in name:
        name, abl = name.split("@", 1)
        abl = abl.replace("+", ",")
    elif name.endswith("_wm"):
        name, abl = name[:-3], "weights,matmul"
    elif name.endswith("_w"):
        name, abl = name[:-2], "weights"
    elif name.endswith("_m"):
        name, abl = name[:-2], "matmul"
    pkg, knobs = VARIANTS[name]
    return pkg, knobs, abl


def apply_variant(variant, monkeypatch):
    global op, pd
    pkg, knobs, abl = _variant(variant)
    op, pd = _MODS[pkg]
    if hasattr(pd, "WSL_SPARE_MODE"):
        for k, v in {**KNOB_DEFAULTS, **knobs}.items():
            monkeypatch.setattr(pd, k, v)
        pd.WSL_LAST.clear()
    else:
        assert not knobs, f"{variant}: {pkg} has no WSL knobs"
    if abl:
        monkeypatch.setenv("CONV2D_3X3_ABLATE", abl)
    else:
        monkeypatch.delenv("CONV2D_3X3_ABLATE", raising=False)
    for name in dir(pd):
        m = getattr(pd, name)
        if callable(m) and hasattr(m, "cache_clear") and name != "_round0_reader_rows":
            m.cache_clear()
    return bool(abl)


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for i, c in enumerate([c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]):
        out.setdefault(c["extras"]["layer"].split(" ")[0].replace(".", "_"), c)
        out[f"focus{i}"] = c
    return out


FOCUS = _focus_cases()
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
# BLOCK shapes of test_conv2d_3x3_block_sharded_perf, plus a few non-Panoptic ring shapes.
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
    # Not Panoptic: 2 channel slices (spare groups of 2 cores), C_out 32 (one owner), d 8.
    "plain_k2_d8": (((32, 64, 64), 32, 8, 8), [256, 32], (2, 8), True, "LoFi", 1),
    # Not Panoptic: 4 channel slices, C_out 128, LoFi.
    "plain_k4": (((32, 64, 128), 128, 1, 1), [256, 32], (4, 8), True, "LoFi", 1),
    # res4.x at LoFi without bias / ReLU.
    "res4_lofi_plain": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), False, "LoFi", 1),
}


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("WSL_CASES", "res4_{1__5}_conv2").split(";") if c]:
        if c == "domain":
            out += DOMAIN
        elif c == "focus22":
            out += [f"focus{i}" for i in range(22)]
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("WSL_VARIANTS", "base,copy").split(",") if v]
PARAMS = []
_i = 0
for rep in range(int(os.environ.get("WSL_REPEAT", "1"))):
    for c in SEL_CASES:
        flip = (os.environ.get("WSL_ORDER") == "alt" and _i % 2 == 1) != (rep % 2 == 1)
        vs = list(reversed(SEL_VARIANTS)) if flip else SEL_VARIANTS
        PARAMS += [(c, v, rep) for v in vs]
        _i += 1
RUN_LOG = HERE / "runs" / os.environ.get("WSL_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
NOCHECK = os.environ.get("WSL_NOCHECK") == "1"


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
            groups=b.num_m_groups,
            block_rows=getattr(g, "block_rows", 0),
            sb=list(pd._ring_subblock(b.block_m_tiles, g.out_n_tiles)),
            remote=bool(g.remote_output),
            m_ranges=[n for _, n in g.m_ranges],
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant,rep", PARAMS, ids=[f"{c}-{v}-{r}" for c, v, r in PARAMS])
def test_wsl(case_id, variant, rep, device, monkeypatch):
    import torch

    ablated = apply_variant(variant, monkeypatch)
    device.disable_and_clear_program_cache()
    info = dict(case=case_id, variant=variant, rep=rep)
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold, in_mc = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold, in_mc = _run_focus(FOCUS[case_id], device)
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
        ablated=ablated,
    )
    info["wsl"] = dict(getattr(pd, "WSL_LAST", {}))
    if not ablated:
        outs = HERE / "runs" / "outs"
        outs.mkdir(parents=True, exist_ok=True)
        torch.save(out.to(torch.bfloat16), outs / f"{case_id.replace(':', '_')}-{variant}.pt")
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not NOCHECK and not ablated:
        assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
