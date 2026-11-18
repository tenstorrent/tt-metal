# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: one-tile-wide tap tilize of the BLOCK N-slice prelude, batched per DEST section (perf experiment).

Every variant runs on this dir's op copy (op_copy/: e40c435b625 + ablation_r7 hooks + this experiment's knobs), whose
knobs default to the committed op's behaviour (variant `base`). Never profile it in one process with the real op
(zone-name hash collisions).

Env: NSNT_CASES (comma list of focus ids or `block:<layer>`), NSNT_VARIANTS (comma list), NSNT_ORDER=alt (reverse the
variant order on every other case), NSNT_LOG (run-log name). One dispatch per test; each appends a JSON record
(blocking, PCC, output hash) to runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp rows in call
order. The precision contract is the case's (LoFi / HiFi4, fp32 DEST, bf8_b): no variant touches it.
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
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_narrow_tilize.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / "op_copy", pd.KERNEL_DIR

FZ = ("NSNT_FINE_ZONES", "1")
VARIANTS = {
    "base": (),
    "fz": (FZ,),
    **{f"b{b}": (("NSNT_BATCH", str(b)),) for b in (2, 3, 4)},
    **{f"b{b}fz": (("NSNT_BATCH", str(b)), FZ) for b in (2, 3, 4)},
    **{f"b{b}p1": (("NSNT_BATCH", str(b)), ("NSNT_PACK_MODE", "1")) for b in (2, 4)},
    # diagnosis (wrong output by design): the reader plans / publishes every unit but issues no gather read
    "nrfz": (("NSNT_DIAG_NOREADS", "1"), FZ),
    "nr": (("NSNT_DIAG_NOREADS", "1"),),
    # diagnosis (wrong output by design): no pack in the batched path
    **{f"b{b}np": (("NSNT_BATCH", str(b)), ("NSNT_DIAG_NOPACK", "1")) for b in (1, 2, 4)},
}


def apply_variant(variant, monkeypatch):
    monkeypatch.setattr(pd, "NSNT_DEFINES", VARIANTS[variant])
    pd.select_blocking.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_").replace("{", "").replace("}", "")] = c
    return out


FOCUS = _focus_cases()
# The BLOCK guard set of tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py (copied, not imported).
# (((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
BLOCK_SHAPES = {
    "res4_0_conv2": (((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    "res4_conv2": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    "res4_conv2_lofi": (((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "LoFi", 1),
}

SEL_CASES = [c for c in os.environ.get("NSNT_CASES", "res4_1__5_conv2").split(",") if c]
SEL_VARIANTS = [v for v in os.environ.get("NSNT_VARIANTS", "base").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("NSNT_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("NSNT_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
ABLATION = bool(os.environ.get("CONV2D_3X3_ABLATE"))


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


def _record_blocking(monkeypatch, info):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            nkb=b.num_k_blocks,
            nslice=bool(b.nslice),
            ring=b.im2col_rm_depth_rows,
            tap_direct=bool(pd._tap_direct(g, b.block_m_tiles, b.block_k_tiles, b.nslice)),
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_nsnt(case_id, variant, device, monkeypatch):
    import torch

    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant, ablate=os.environ.get("CONV2D_3X3_ABLATE", ""))
    _record_blocking(monkeypatch, info)
    if case_id.startswith("block:"):
        tt_output, ref, pcc_threshold = _run_block(case_id.split(":", 1)[1], device)
    else:
        tt_output, ref, pcc_threshold = _run_focus(FOCUS[case_id], device)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        mae=float((out - ref.float()).abs().mean()),
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not ABLATION:
        assert info["finite"] and passed, info
