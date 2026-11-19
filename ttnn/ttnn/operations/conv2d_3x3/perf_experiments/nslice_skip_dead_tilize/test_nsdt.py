# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: skip the dead units of the BLOCK N-slice padded-slab prelude tilize (perf experiment
`nslice_skip_dead_tilize`).

Every variant runs on this dir's op copy (op_copy/: conv2d_3x3 at a51305feccb + the knob NSDT_SKIP_DEAD, default off =
the committed op, variant `base`). Never profile it in one process with the real op (zone-name hash collisions).

Env: NSDT_CASES (comma list of focus ids or `block:<layer>`), NSDT_VARIANTS (comma list), NSDT_ORDER=alt (reverse the
variant order on every other case), NSDT_LOG (run-log name), NSDT_COPY_DIR (op copy dir, default op_copy). One dispatch
per test; each appends a JSON record (path, PCC, output hash) to runs/<log>.jsonl, which report.py joins with the
profiler CSV's GenericOp rows in call order. The precision contract is the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no
variant touches it.
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
_COPY = os.environ.get("NSDT_COPY_DIR", "op_copy")
_PKG = f"ttnn.operations.conv2d_3x3.perf_experiments.nslice_skip_dead_tilize.{_COPY}"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / _COPY, pd.KERNEL_DIR

# variant -> (NSDT_SKIP_DEAD, NSDT_MODEL, NSDT_CENTER)
VARIANTS = {
    "base": (False, False, False),
    # NSDT_SKIP_DEAD 1: a per-unit test inside the kernel-row loop.
    "skip": (True, False, False),
    "skipm": (True, True, False),
    # NSDT_SKIP_DEAD 2: the row-range form (dead prefix / suffix advanced once per row, base loop over the live range).
    "skipr": (2, False, False),
    "skiprm": (2, True, False),
    # NSDT_SKIP_DEAD 3: the row-range form only on cores with a dead prelude unit; elsewhere the committed loop.
    "skipd": (3, False, False),
    "skipdm": (3, True, False),
    # skipd + the reader writes no center zero row the walk never reads.
    "skipdc": (3, False, True),
}


def apply_variant(variant, monkeypatch):
    skip, model, center = VARIANTS[variant]
    # grad_copy (the graduation candidate) has no knobs: its `base` is the candidate itself.
    monkeypatch.setattr(pd, "NSDT_SKIP_DEAD", skip, raising=False)
    monkeypatch.setattr(pd, "NSDT_MODEL", model, raising=False)
    monkeypatch.setattr(pd, "NSDT_CENTER", center, raising=False)
    for fn in (pd.select_blocking, pd.balanced_m_ranges, pd.uniform_range_cost, pd.balanced_block_ranges):
        fn.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_").replace("{", "").replace("}", "")] = c
    return out


FOCUS = _focus_cases()
# The BLOCK set of tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py (copied, not imported).
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
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

SEL_CASES = [c for c in os.environ.get("NSDT_CASES", "aspp_convs_3").split(",") if c]
SEL_VARIANTS = [v for v in os.environ.get("NSDT_VARIANTS", "base,skip").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("NSDT_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("NSDT_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


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
        torch_input.float(),
        torch_weight.float(),
        input_height=H,
        input_width=W,
        bias=None if torch_bias is None else torch_bias.float(),
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    )
    pcc = (extras.get("tolerance") or (0.99, None))[0]
    return tt_output, ref, pcc


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
    assert tt_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
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
    return tt_output, ref, 0.99


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_nsdt(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold = _run_focus(FOCUS[case_id], device)
    in_mem = tt_output.memory_config()
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        path=dict(getattr(pd, "LAST_PATH_INFO", {})),
        out_layout=str(in_mem.memory_layout),
        out_shard=list(in_mem.shard_spec.shape) if in_mem.shard_spec is not None else None,
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        mae=float((out - ref.float()).abs().mean()),
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert info["finite"] and passed, info
