# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op A/B on a private copy of conv2d_3x3 (op_copy/) for the round-5 idea `nslice_paired_weight_lead`: run the
N-slice-outer walk at paired-subblock speed and remove what gates it (the weight stream).

    LO_CASES=res5_1_conv2,res5_2_conv2 LO_VARIANTS=baseline,lead LO_LOG=prof1 \\
        scripts/run_safe_pytest.sh --profile --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_paired_weight_lead/test_lead_op.py --import-mode=prepend
    python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_paired_weight_lead/report.py <ops_perf_results.csv>

Variants (all on the copy, so the instrumentation is identical; never profile the copy next to the real op):
  baseline   the committed op (2ac42333eb2): one helper matmul per owner slice, weight depth 4, writer-paced rounds.
  lead       weight lead: every weight round of the kernel resident (depth = rounds), the column's multicast
             handshake-free with a Counter data-ready signal, the writer pushing rounds and sending slices from one
             polling loop (see CONV2D_NSLICE_LEAD in the copy's writer).
  tall       raw per-slice walk in 4-row-tall subblocks (rt 4 / 3 x ct 1): the paired subblock's unpack ratio without
             touching the weight layout (see CONV2D_NSLICE_WALK in the copy's compute).
  pair       pair rounds: two owner slices' weights interleaved per K row ([K, 2 out_n]) and one 1 x 2 out_n matmul.
  lead_tall, lead_pair: the combinations.

Cases: the perf-focus LOOSE cases (ids as in test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus) and the BLOCK
harness shapes (test_conv2d_3x3_block_sharded_perf, ids prefixed `blk_`). The precision contract is each case's own
(fp32 DEST, its fidelity, bf8_b): every variant runs it unchanged. Each dispatch appends a JSON record (blocking, PCC,
output hash) to runs/<LO_LOG>.jsonl; report.py joins it with the profiler CSV's GenericOp rows in call order.
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
# No __init__.py anywhere under perf_experiments (ttnn walk_packages would execute it on `import ttnn`).
# LO_PKG=op_patch runs graduation.patch's tree (the base op with the patch applied; no knobs) instead of op_copy.
_SUB = os.environ.get("LO_PKG", "op_copy")
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_paired_weight_lead." + _SUB
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / _SUB, pd.KERNEL_DIR

KNOB_DEFAULTS = dict(
    NSLICE_LEAD=False, NSLICE_WALK="slice", NSLICE_LEAD_READS_AHEAD=2, NSLICE_WALK_ZONES=False, NSLICE_SLICE_DEPTH_MIN=0
)
VARIANTS = {
    "baseline": {},
    "lead": dict(NSLICE_LEAD=True),
    "lead_r3": dict(NSLICE_LEAD=True, NSLICE_LEAD_READS_AHEAD=3),
    "tall": dict(NSLICE_WALK="tall"),
    "pair": dict(NSLICE_WALK="pair"),
    "lead_tall": dict(NSLICE_LEAD=True, NSLICE_WALK="tall"),
    "lead_pair": dict(NSLICE_LEAD=True, NSLICE_WALK="pair"),
    # Diagnostics: per-slice compute zones (weight wait, slice-CB reserve, matmul).
    "base_z": dict(NSLICE_WALK_ZONES=True),
    "tall_z": dict(NSLICE_WALK="tall", NSLICE_WALK_ZONES=True),
    # Two-deep slice CBs on the shared-L1 walk (from the L1 left after the solve).
    "tall_sd2": dict(NSLICE_WALK="tall", NSLICE_SLICE_DEPTH_MIN=2),
    "tall_sd2_z": dict(NSLICE_WALK="tall", NSLICE_SLICE_DEPTH_MIN=2, NSLICE_WALK_ZONES=True),
    "lead_tall_sd2": dict(NSLICE_LEAD=True, NSLICE_WALK="tall", NSLICE_SLICE_DEPTH_MIN=2),
}


def apply_variant(variant, monkeypatch):
    for k, v in {**KNOB_DEFAULTS, **VARIANTS[variant]}.items():
        if hasattr(pd, k):
            monkeypatch.setattr(pd, k, v)
        else:
            assert v == KNOB_DEFAULTS[k], f"{_SUB} has no knob {k}"
    pd.select_blocking.cache_clear()


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = ("focus", c)
    return out


# Mirror of test_conv2d_3x3_perf.py BLOCK_SHAPES:
# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
BLOCK_SHAPES = [
    ("res4_0_conv2", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, "HiFi4", 2),
    ("res4_conv2", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "HiFi4", 1),
    ("res4_conv2_lofi", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, "LoFi", 1),
    ("res5_0_conv2", ((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, "HiFi4", 1),
    ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, "HiFi4", 1),
    ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, "HiFi4", 1),
    ("aspp_convs1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, "HiFi4", 1),
    ("aspp_convs2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, "HiFi4", 1),
    ("aspp_convs3", ((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, "HiFi4", 1),
]

CASES = {**_focus_cases(), **{"blk_" + s[0]: ("block", s) for s in BLOCK_SHAPES}}
FOCUS = ["res5_1_conv2", "res5_2_conv2"]
SEL_CASES = [c for c in os.environ.get("LO_CASES", ",".join(FOCUS)).split(",") if c]
if SEL_CASES == ["all"]:
    SEL_CASES = list(CASES)
SEL_VARIANTS = [v for v in os.environ.get("LO_VARIANTS", "baseline,lead").split(",") if v]
RUN_LOG = HERE / "runs" / (os.environ.get("LO_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}") + ".jsonl")


def run_focus(case, device):
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
    pcc_threshold = (extras.get("tolerance") or (0.99, None))[0]
    return tt_output, ref, pcc_threshold


def run_block(shape, device):
    import torch

    layer, case, shard_shape, (gx, gy), bias_relu, fidelity, stride = shape
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, dilation = case
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
            nslice=bool(b.nslice),
            lead=bool(getattr(b, "lead", False)),
            walk=getattr(b, "walk", "slice"),
            out_n=g.out_n_tiles,
            k_cols=g.k_cols,
            k_tiles=g.tensor_k_tiles,
            m_blocks=b.max_m_blocks_per_core,
            weight_depth=b.weight_depth,
            slice_depth=b.slice_depth,
            alias=b.shared_slab_bytes > 0,
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)


@pytest.mark.parametrize("variant", SEL_VARIANTS)
@pytest.mark.parametrize("case_id", SEL_CASES)
def test_lead_op(case_id, variant, device, monkeypatch):
    import torch

    kind, case = CASES[case_id]
    apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant if _SUB == "op_copy" else f"{_SUB}:{variant}")
    _record_blocking(monkeypatch, info)
    tt_output, ref, pcc_threshold = run_focus(case, device) if kind == "focus" else run_block(case, device)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref.float(), out, pcc_threshold)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    assert info["finite"] and passed, info
