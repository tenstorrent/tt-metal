# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: run the BLOCK K split on more grid rows than the input's shard grid (perf experiment block_full_grid).

Every variant runs on this dir's op copy (op_copy/: 5f23b256d1a + the BFG_* knobs). Never profile it in one process
with the real op (zone-name hash collisions).

Variants (knobs in op_copy/conv2d_3x3_program_descriptor.py):
  baseline  BFG_EXTEND_ROWS=False: the op as committed (M split down the shard rows, output packed in place).
  extend    BFG_EXTEND_ROWS=True: the M split runs down the fewest grid rows (to the device edge) that lower the
            output tile-rows per row; compute packs into a local CB and the writer NoC-writes each finished tile to
            its output shard.
  spare     extend + BFG_SPARE_COLS: more M-groups, transposed into the device columns past the shard grid (one group
            of k_cols cores down a spare column), streaming their own weights.

Env: BFG_CASES (focus ids, `block:<layer>`, or `focus` / `block` / `all`), BFG_VARIANTS (comma list), BFG_ORDER=alt
(reverse the variant order on every other case), BFG_LOG. One dispatch per test, for `run_safe_pytest.sh --profile`;
each dispatch appends a JSON record (blocking, PCC, output hash) to runs/<log>.jsonl, which report.py joins with the
profiler CSV's GenericOp rows in call order. Precision contract fixed per case (its fidelity, fp32 DEST, its dtypes).
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
# BFG_PKG=grad runs the graduation copy (grad/: the op at 5f23b256d1a + graduation.patch) instead, variant `grad` only.
_COPY = os.environ.get("BFG_PKG", "op_copy")
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.block_full_grid." + _COPY
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / _COPY, pd.KERNEL_DIR

KNOB_DEFAULTS = dict(BFG_EXTEND_ROWS=False, BFG_SPARE_COLS=False)
VARIANTS = {
    "baseline": {},
    "extend": dict(BFG_EXTEND_ROWS=True),
    "spare": dict(BFG_EXTEND_ROWS=True, BFG_SPARE_COLS=True),
}


def apply_variant(variant, monkeypatch):
    if _COPY == "grad":
        assert variant == "grad", "the graduation copy has no knobs"
    else:
        for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
            monkeypatch.setattr(pd, k, v)
    # select_blocking is lru_cached on the geometry only: a knob must not hit a cached solve.
    pd.select_blocking.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = c
    return out


FOCUS = _focus_cases()
# The BLOCK guard set of tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py (copied, not imported).
# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
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
}
FOCUS_DEFAULT = ["aspp_convs_3", "res4_{1__5}_conv2", "aspp_convs_2", "aspp_convs_1", "res4_0_conv2"]


def _sel_cases():
    raw = [c for c in os.environ.get("BFG_CASES", "aspp_convs_3").split(",") if c]
    out = []
    for c in raw:
        if c == "focus":
            out += FOCUS_DEFAULT
        elif c == "block":
            out += [f"block:{k}" for k in BLOCK_SHAPES]
        elif c == "all":
            out += FOCUS_DEFAULT + [f"block:{k}" for k in BLOCK_SHAPES]
        else:
            out.append(c)
    return out


SEL_CASES = _sel_cases()
SEL_VARIANTS = [v for v in os.environ.get("BFG_VARIANTS", "baseline").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("BFG_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("BFG_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


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
    return tt_input, tt_output, ref, pcc


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
    return tt_input, tt_output, ref, 0.99


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            rows=grid[0],
            grid_rows=getattr(g, "block_rows", None),
            remote=bool(g.remote_output),
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            core_m=b.core_m_tiles,
            nslice=bool(b.nslice),
            ub=b.untilize_batch_rows,
            slab=b.slab_tile_rows,
            mblk=b.max_m_blocks_per_core,
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_bfg(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_input, tt_output, ref, pcc_threshold = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_input, tt_output, ref, pcc_threshold = _run_focus(FOCUS[case_id], device)
    # The output keeps exactly the input's shard grid, orientation, and the shard rows' split.
    in_spec, out_spec = tt_input.memory_config().shard_spec, tt_output.memory_config().shard_spec
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert out_spec.grid == in_spec.grid and out_spec.orientation == in_spec.orientation
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        out_shard=list(out_spec.shape),
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        max_abs=float((out - ref.float()).abs().max()),
        mean_abs=float((out - ref.float()).abs().mean()),
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert info["finite"] and passed, info
