# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: work-balanced M split of the BLOCK K split (perf experiment, round 8, `block_balanced_msplit`).

Every variant runs on this dir's op copy (op_copy/: 3ca4eb28f46 + this experiment's knobs), whose knobs default to the
committed op's behaviour (variant `baseline`). Never profile it in one process with the real op (zone-name hash
collisions).

Env: BBM_CASES (focus ids or `block:<layer>`, comma list; `domain` = the BLOCK focus set), BBM_VARIANTS (comma list,
run in that order per case), BBM_ORDER=alt (reverse the order on every other case), BBM_LOG (run-log name). One dispatch
per test, for `run_safe_pytest.sh --profile`; each dispatch appends a JSON record (blocking, M split, PCC, mean abs
error vs a fp32 torch conv, output hash) to runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp
rows in call order. The precision contract is the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it.
Inputs are seeded per case, so every variant's output hash compares with the baseline's.
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
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.block_balanced_msplit.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / "op_copy", pd.KERNEL_DIR

# BALANCED_MSPLIT: the host's work-balanced M split of the K split's grid rows (and the kernels' runtime valid-row
# counts it needs). BALANCED_ROWS: candidate row counts (None: block_compute_rows' count only).
KNOB_DEFAULTS = dict(BALANCED_MSPLIT=False, RUNTIME_VALID_ROWS=False, WALK_CT_GROUP_ROWS=False)
VARIANTS = {
    "baseline": {},
    # The base uniform split, with the kernels sizing the prelude / walk / combine by the core's valid tile-rows.
    "rtvalid": dict(RUNTIME_VALID_ROWS=True),
    # The work-balanced split (implies the runtime valid rows).
    "balanced": dict(BALANCED_MSPLIT=True, RUNTIME_VALID_ROWS=True),
    # The ring walk's group matmuls specialized per group height (uniform split).
    "ctrows": dict(WALK_CT_GROUP_ROWS=True),
    # Both, forced on every split.
    "bal_ct": dict(BALANCED_MSPLIT=True, RUNTIME_VALID_ROWS=True, WALK_CT_GROUP_ROWS=True),
    # The graduation candidate: the balanced split where it is taken, with runtime sizing and template group heights
    # only there; the uniform split otherwise runs the base kernels' path.
    "final": dict(BALANCED_MSPLIT=True),
}
_EXTRA = os.environ.get("BBM_KNOBS")  # e.g. "BALANCE_COST_A=1.0;BALANCE_COST_B=2.0" applied to every non-baseline
if _EXTRA:
    for name in list(VARIANTS):
        if name != "baseline":
            for kv in _EXTRA.split(";"):
                k, v = kv.split("=")
                VARIANTS[name] = dict(VARIANTS[name], **{k: eval(v)})


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        if hasattr(pd, k):
            monkeypatch.setattr(pd, k, v)
    if hasattr(pd.select_blocking, "cache_clear"):
        pd.select_blocking.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = c
    return out


FOCUS = _focus_cases()
DOMAIN = [
    "aspp_convs_3",
    "aspp_convs_1",
    "aspp_convs_2",
    "res5_0_conv2",
    "res5_1_conv2",
    "res5_2_conv2",
    "res4_{1__5}_conv2",
    "res4_0_conv2",
]
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
BLOCK_SHAPES = {
    "res4_0_conv2": (((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    "res4_conv2": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    "res5_0_conv2": (((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, "HiFi4", 1),
    "res5_2_conv2": (((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, "HiFi4", 1),
    "aspp_convs1": (((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, "HiFi4", 1),
    "aspp_convs3": (((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, "HiFi4", 1),
    # Not Panoptic layers. W % 32 != 0 (a tile-row spans two output rows), d = 12, a ragged last row.
    "plain_w48_d12": (((32, 48, 1024), 256, 12, 12), [192, 96], (11, 8), False, "LoFi", 1),
    # stride 2 (strided-window prelude), d = 8.
    "plain_s2_d8": (((64, 128, 512), 256, 8, 8), [1024, 64], (8, 8), True, "LoFi", 2),
    # aspp.convs.3-like with no ragged grid row (80 tile-rows on 10 rows of 8): runtime-sized kernels do the base work.
    "plain_h40_d18": (((40, 64, 1024), 256, 18, 18), [320, 96], (11, 8), False, "LoFi", 1),
    # bias, out_n 2, res5-like grid with d = 6 and C_out 384 (a short last N-slice), all rows in the shard grid.
    "plain_res5_d6": (((32, 64, 512), 384, 6, 6), [224, 64], (8, 10), True, "LoFi", 1),
}


def _sel_cases():
    out = []
    for c in [c for c in os.environ.get("BBM_CASES", "aspp_convs_3").split(",") if c]:
        if c == "domain":
            out += DOMAIN
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("BBM_VARIANTS", "baseline,balanced").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("BBM_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("BBM_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


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
        info["ranges"] = [list(r) for r in g.m_ranges]
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            nkb=b.num_k_blocks,
            nslice=bool(b.nslice),
            core_m=b.core_m_tiles,
            groups=b.num_m_groups,
            k_cols=g.k_cols,
            remote=bool(g.remote_output),
            slab=b.slab_tile_rows,
            fp=b.footprint_bytes,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_bbm(case_id, variant, device, monkeypatch):
    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold, in_mc = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold, in_mc = _run_focus(FOCUS[case_id], device)
    # The output keeps exactly the input's shard grid / orientation and the base op's shard shape.
    out_mc = tt_output.memory_config()
    info["out_shard"] = [list(out_mc.shard_spec.shape), str(out_mc.shard_spec.grid), str(out_mc.shard_spec.orientation)]
    assert out_mc.memory_layout == in_mc.memory_layout
    assert str(out_mc.shard_spec.grid) == str(in_mc.shard_spec.grid)
    assert out_mc.shard_spec.orientation == in_mc.shard_spec.orientation
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        mae=float((out - ref.float()).abs().mean()),
        out_sha=hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert passed, f"{case_id} {variant}: pcc {pcc} < {pcc_threshold}"
