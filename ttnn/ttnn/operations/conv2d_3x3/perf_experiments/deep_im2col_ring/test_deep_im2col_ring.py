# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: deep cb_im2col_rm ring on conv2d_3x3's K-blocked walk (perf experiment, not part of the op).

The K-blocked walk tilizes K-block k (compute's PreKBlockFn) right before multiplying it. With cb_im2col_rm at
IM2COL_RM_DEPTH_ROWS = 2 tile-rows, the reader gathers only 2 tile-rows of K-block k+1 during matmul k, then blocks
on cb_reserve_back; the tilize of k+1 then waits on the gather of every remaining tile-row. The candidate makes the
ring depth a solve output: after (bm, bk, untilize batch) are chosen at depth 2, the ring grows toward a target as far
as the L1 left over allows (op_copy/conv2d_3x3_program_descriptor.py: IM2COL_RM_DEEP, _deep_ring_target, solve()).
No kernel changes: on the non-split gather the ring depth is only a CB size.

Variants (DIR_VARIANTS, comma-separated; one dispatch per (case, variant), in that order; all on this dir's copy):
  d2     baseline: IM2COL_RM_DEEP = None (the committed op's fixed depth 2; the copy is otherwise identical)
  bm     target one K-block of tile-rows (block_m_tiles)
  2bm    target two K-blocks
  dN     target N tile-rows

Cases (DIR_CASES, comma-separated):
  focus:<id>           a perf-flagged feature_spec LOOSE case at its exact config (id as in the focus harness)
  height:<layer>       a HEIGHT perf-harness layer (test_conv2d_3x3_perf.HEIGHT_SHAPES)
  block:<layer>        a BLOCK perf-harness layer (test_conv2d_3x3_perf.BLOCK_SHAPES)
  inter:<shape>:<lay>  an interleaved perf-harness shape; lay in rm_bf16 | tile_bf16 | tile_bf8b
  height_all / inter_all / focus_all / block_all expand to every case of that harness.

Every dispatch is correctness-gated: PCC vs torch at the case's threshold and bit-equality with the d2 output of the
same case in the same pass (when d2 ran first). Precision contract (fp32 DEST, fidelity, approx mode, dtypes) is the
case's own and identical in every variant.

Run:
  DIR_CASES=focus:semantic DIR_VARIANTS=d2,bm,2bm scripts/run_safe_pytest.sh --profile --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/deep_im2col_ring/test_deep_im2col_ring.py
  python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/deep_im2col_ring/report.py <ops_perf_results csv>
"""

import importlib
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
# No __init__.py anywhere in this experiment (ttnn walk_packages-executes every module under a package on
# `import ttnn`): the op copy is mounted as a synthetic package so its relative imports resolve to the copy.
OPC = "deep_im2col_ring_op_copy"
if OPC not in sys.modules:
    _pkg = types.ModuleType(OPC)
    _pkg.__path__ = [str(HERE / "op_copy")]
    sys.modules[OPC] = _pkg
pd = importlib.import_module(OPC + ".conv2d_3x3_program_descriptor")
op = importlib.import_module(OPC + ".conv2d_3x3")
perf = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf")

VARIANTS = {"d2": None, "bm": "bm", "2bm": "2bm"}


def _variant_target(name):
    if name in VARIANTS:
        return VARIANTS[name]
    assert name.startswith("d") and name[1:].isdigit(), f"bad variant {name}"
    return int(name[1:])


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    return {
        c["extras"]["layer"].split(" ")[0].replace(".", "_"): c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]
    }


LAYOUT_IDS = {
    "rm_bf16": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    "tile_bf16": (ttnn.bfloat16, ttnn.TILE_LAYOUT),
    "tile_bf8b": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
}


def _expand(tokens):
    out = []
    for t in tokens:
        if t == "height_all":
            out += [f"height:{c[0]}" for c in perf.HEIGHT_SHAPES]
        elif t == "block_all":
            out += [f"block:{c[0]}" for c in perf.BLOCK_SHAPES]
        elif t == "focus_all":
            out += [f"focus:{k}" for k in _focus_cases()]
        elif t == "inter_all":
            out += [f"inter:{perf._shape_id(s)}:{lay}" for s in perf.PERF_SHAPES for lay in LAYOUT_IDS]
        else:
            out.append(t)
    return out


DIR_CASES = _expand([c for c in os.environ.get("DIR_CASES", "focus:semantic").split(",") if c])
DIR_VARIANTS = [v for v in os.environ.get("DIR_VARIANTS", "d2,bm").split(",") if v]
LOG = HERE / "runs" / f"run_{int(time.time())}_{os.getpid()}.jsonl"
_baseline_out = {}
_runs = {}


def _build(case_token, device):
    """(spec dict, torch_input, torch_weight, torch_bias) for one case token; spec carries everything the call needs."""
    import torch

    kind, _, rest = case_token.partition(":")
    torch.manual_seed(0)
    if kind == "focus":
        from eval.golden_tests.conv2d_3x3 import helpers

        case = _focus_cases()[rest]
        (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
        extras = case["extras"]
        dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
        mem = helpers.build_input_memory_config(
            (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
        )
        if mem.shard_spec is None or list(mem.shard_spec.shape) != list(extras["shard_shape"]):
            pytest.skip(f"{rest}: pinned shard geometry does not fit this device's grid")
        has_bias = case["bias_mode"] == "with_bias"
        activation = None if case["activation"] == "none" else case["activation"]
        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
        cfg = helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        )
        pcc = extras.get("tolerance", (0.99, None))[0]
        torch_input = torch_input.reshape(1, 1, H * W, c_in)
    elif kind == "height":
        layer, case, dtype, layout, shard_shape, num_cores = next(c for c in perf.HEIGHT_SHAPES if c[0] == rest)
        (H, W, c_in), c_out, padding, stride = case
        dilation, weight_dtype, activation = 1, ttnn.bfloat8_b, "relu"
        grid = device.compute_with_storage_grid_size()
        if num_cores > grid.x * grid.y or grid.x < 11:
            pytest.skip(f"{layer}: pinned grid does not fit")
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(perf._row_major_cores(num_cores, 11), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16)
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        cfg, pcc = None, 0.99
    elif kind == "block":
        layer, case, shard_shape, (gx, gy), bias_relu, fidelity, stride = next(
            c for c in perf.BLOCK_SHAPES if c[0] == rest
        )
        (H, W, c_in), c_out, padding, dilation = case
        dtype, layout, weight_dtype = ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.bfloat8_b
        activation = "relu" if bias_relu else None
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
                shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if bias_relu else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)
        pcc = 0.99
    elif kind == "inter":
        shape_id, _, lay = rest.partition(":")
        (H, W, c_in), c_out, padding = next(s for s in perf.PERF_SHAPES if perf._shape_id(s) == shape_id)
        dtype, layout = LAYOUT_IDS[lay]
        stride, dilation, weight_dtype, activation = 1, 1, ttnn.bfloat16, None
        mem = ttnn.DRAM_MEMORY_CONFIG
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16)
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        cfg, pcc = None, (0.995 if dtype == ttnn.bfloat16 else 0.99)
    else:
        raise ValueError(case_token)
    spec = dict(
        H=H,
        W=W,
        c_in=c_in,
        c_out=c_out,
        padding=padding,
        stride=stride,
        dilation=dilation,
        dtype=dtype,
        layout=layout,
        weight_dtype=weight_dtype,
        mem=mem,
        activation=activation,
        cfg=cfg,
        pcc=pcc,
    )
    return spec, torch_input, torch_weight, torch_bias


@pytest.mark.parametrize("variant", DIR_VARIANTS)
@pytest.mark.parametrize("case_token", DIR_CASES)
def test_deep_im2col_ring(case_token, variant, device, monkeypatch):
    import torch

    spec, torch_input, torch_weight, torch_bias = _build(case_token, device)
    monkeypatch.setattr(pd, "IM2COL_RM_DEEP", _variant_target(variant))
    pd.select_blocking.cache_clear()
    pd._block_extents.cache_clear()
    select = pd.select_blocking
    blockings = []

    def recording_select(g, grid, l1_budget):
        b = select(g, grid, l1_budget)
        blockings.append(
            dict(
                bm=b.block_m_tiles,
                bk=b.block_k_tiles,
                nk=b.num_k_blocks,
                mblocks=b.max_m_blocks_per_core,
                ub=b.untilize_batch_rows,
                depth=b.im2col_rm_depth_rows,
                nslice=b.nslice,
                row_split=pd._row_split(g, b.block_m_tiles, b.block_k_tiles),
                face_direct=pd.face_direct_im2col(g, b.block_k_tiles),
                tap_major=pd.tap_major_im2col(g, b),
                footprint=b.footprint_bytes,
                budget=l1_budget,
            )
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording_select)

    kwargs = dict(
        stride=spec["stride"],
        padding=spec["padding"],
        dilation=spec["dilation"],
        input_dtype=spec["dtype"],
        input_layout=spec["layout"],
        weight_dtype=spec["weight_dtype"],
        input_memory_config=spec["mem"],
        device=device,
    )
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=spec["dtype"], layout=spec["layout"], device=device, memory_config=spec["mem"]
    )
    call_kwargs = dict(
        input_height=spec["H"],
        input_width=spec["W"],
        bias=tt_bias,
        padding=spec["padding"],
        stride=spec["stride"],
        dilation=spec["dilation"],
        activation=spec["activation"],
    )
    if spec["cfg"] is not None:
        call_kwargs["compute_kernel_config"] = spec["cfg"]
    tt_output = op.conv2d_3x3(tt_input, tt_weight, **call_kwargs)
    c_out = spec["c_out"]
    out = ttnn.to_torch(tt_output).float().reshape(1, 1, -1, c_out)

    H, W, c_in = spec["H"], spec["W"], spec["c_in"]
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=spec["stride"],
        padding=spec["padding"],
        dilation=spec["dilation"],
    )
    if spec["activation"] == "relu":
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    passed, pcc = comp_pcc(ref, out, spec["pcc"])
    # A precompile pass may run every body once before the real pass: compare only outputs of the same pass.
    pass_idx = _runs[(case_token, variant)] = _runs.get((case_token, variant), 0) + 1
    if variant == "d2":
        _baseline_out[(case_token, pass_idx)] = out
    equal = None
    if variant != "d2" and (case_token, pass_idx) in _baseline_out:
        equal = bool(torch.equal(out, _baseline_out[(case_token, pass_idx)]))
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(
            json.dumps(
                dict(
                    case=case_token,
                    variant=variant,
                    pcc=float(pcc),
                    equal_d2=equal,
                    blocking=blockings[-1] if blockings else None,
                )
            )
            + "\n"
        )
    assert passed, f"{case_token}/{variant}: PCC {pcc} < {spec['pcc']}"
    assert equal is not False, f"{case_token}/{variant}: output differs from the depth-2 baseline"
