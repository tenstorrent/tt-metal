# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the K split's prepared-weight page layout (perf experiment, not part of the op).

On the BLOCK_SHARDED N-slice-outer K split, each grid column's weight-multicast sender reads one owner N-slice per round
(K_col x out_n tiles). With the op's row-major weight pages (page = k * tensor_n_tiles + n, bank = page % 8) one round's
pages all sit in one DRAM bank (aspp: tensor_n_tiles 8) or two (res5: 16 tiles, 2 per slice). Variants, all on this
dir's op copy (op_copy/, the round-3 tree + the ablation_r3 hooks + the layout knob), all bit-identical by construction:

  base    WEIGHT_LAYOUT 0: the op's row-major layout and writer (the honest baseline; the copy is otherwise identical)
  spread  WEIGHT_LAYOUT 1: slice-major page runs inside each column's page range; a round is one run of consecutive
          pages (all 8 banks), still one NoC read per tile
  chunk   WEIGHT_LAYOUT 2: the same runs, permuted so the round's tiles in each bank are one contiguous bank range and one
          contiguous L1 range: one NoC read per bank per round (8 instead of K_col x out_n)

Only the K split changes layout (prepare_conv2d_weights sees the BLOCK shard spec); HEIGHT / interleaved keep row-major
and the writer compiles the identical code there (weight_layout is 0 unless k_split).

Cases (WBS_CASES, comma-separated): focus:<id> | block:<layer> | height:<layer> | inter:<shape_id>:<rm_bf16|tile_bf16|
tile_bf8b> | block_unaligned (a C_out-unaligned BLOCK K split) | focus_all | block_all.
Variants (WBS_VARIANTS, comma-separated, dispatched in that order per case).

Every dispatch is correctness-gated unless CONV2D_3X3_PERF_ABLATION=1 (ablated payloads are wrong by construction):
PCC vs torch at the case's threshold, and torch.equal against the `base` output of the same case in the same process.

Run (one fresh-cache dispatch per (case, variant); the device profiler's DEVICE KERNEL DURATION is the metric):
  WBS_CASES=focus:aspp_convs_1 WBS_VARIANTS=base,spread,chunk scripts/run_safe_pytest.sh --profile --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_bank_spread/test_weight_bank_spread.py \\
      -p no:cacheprovider --import-mode=prepend
  python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_bank_spread/report.py <ops_perf_results csv>
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

HERE = Path(__file__).parent
# No __init__.py anywhere in this experiment (ttnn walk_packages-executes every module under a package on
# `import ttnn`): the op copy is mounted as a synthetic package so its relative imports resolve to the copy.
OPC = "weight_bank_spread_op_copy"
if OPC not in sys.modules:
    _pkg = types.ModuleType(OPC)
    _pkg.__path__ = [str(HERE / "op_copy")]
    sys.modules[OPC] = _pkg
pd = importlib.import_module(OPC + ".conv2d_3x3_program_descriptor")
op = importlib.import_module(OPC + ".conv2d_3x3")
perf = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_perf")

LAYOUTS = {"base": 0, "spread": 1, "chunk": 2}


def _parse_variant(variant):
    """<layout>[_diag][_kb] -> (layout id, sender stagger, forced K-blocked walk, the variant its output must equal).
    _diag: the K split's column multicast senders staggered down a diagonal (McastSenderPlacement.Staggered).
    _kb: skip the N-slice-outer solve (correctness probe of the K-blocked walk's weight reads); its outputs are compared
    with base_kb, since the K-blocked walk accumulates in a different order than the N-slice walk."""
    name, kb = (variant[:-3], True) if variant.endswith("_kb") else (variant, False)
    name, diag = (name[:-5], True) if name.endswith("_diag") else (name, False)
    return LAYOUTS[name], diag, kb, "base_kb" if kb else "base"


ABLATION = os.environ.get("CONV2D_3X3_PERF_ABLATION") == "1"


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
        if t == "block_all":
            out += [f"block:{c[0]}" for c in perf.BLOCK_SHAPES]
        elif t == "focus_all":
            out += [f"focus:{k}" for k in _focus_cases()]
        else:
            out.append(t)
    return out


WBS_CASES = _expand([c for c in os.environ.get("WBS_CASES", "focus:aspp_convs_1").split(",") if c])
WBS_VARIANTS = [v for v in os.environ.get("WBS_VARIANTS", "base,spread,chunk").split(",") if v]
LOG = HERE / "runs" / f"run_{int(time.time())}_{os.getpid()}.jsonl"
_outputs = {}


def _block_mem(gx, gy, shard_shape, x0=0, y0=0):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + gx - 1, y0 + gy - 1))]),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _build(case_token, device):
    """(spec dict, torch_input, torch_weight, torch_bias) for one case token."""
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
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16).reshape(1, 1, H * W, c_in)
        cfg = helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        )
        pcc = extras.get("tolerance", (0.99, None))[0]
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
        mem = _block_mem(gx, gy, shard_shape)
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if bias_relu else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)
        pcc = 0.99
    elif kind == "block_unaligned":
        # C_out 200 (7 tile columns, last one partial) over 3 channel slices (out_n 3: slices 3, 3, 1 tiles, one short):
        # exercises the padded-width view and the short-last-slice chunk split. bf16 TILE, offset grid.
        H, W, c_in, c_out, padding, stride, dilation = 16, 32, 192, 200, 1, 1, 1
        dtype, layout, weight_dtype = ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.bfloat8_b
        activation = "relu"
        mem = _block_mem(3, 4, [128, 64], x0=1, y0=1)
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16)
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
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


def _collecting(request):
    """True inside run_safe_pytest's inline precompile pass, which runs every body once in this process first."""
    try:
        from tests.plugins.up_front_collect import _COLLECTED_BODY_KEY
    except ImportError:
        return False
    return _COLLECTED_BODY_KEY in request.node.stash


@pytest.mark.parametrize("variant", WBS_VARIANTS)
@pytest.mark.parametrize("case_token", WBS_CASES)
def test_weight_bank_spread(case_token, variant, device, monkeypatch, request):
    import torch

    from tests.ttnn.utils_for_testing import comp_pcc

    spec, torch_input, torch_weight, torch_bias = _build(case_token, device)
    layout_id, diag, kb, base_variant = _parse_variant(variant)
    monkeypatch.setattr(pd, "WEIGHT_LAYOUT", layout_id)
    monkeypatch.setattr(pd, "WEIGHT_SENDER_STAGGER", diag)
    monkeypatch.setattr(pd, "FORCE_KBLOCKED", kb)
    pd.select_blocking.cache_clear()
    select = pd.select_blocking
    blockings = []

    def recording_select(g, grid, l1_budget):
        b = select(g, grid, l1_budget)
        blockings.append(
            dict(
                k_cols=g.k_cols,
                T=g.tensor_n_tiles,
                out_n=g.out_n_tiles,
                k_col=g.tensor_k_tiles,
                bm=b.block_m_tiles,
                bk=b.block_k_tiles,
                nk=b.num_k_blocks,
                mblocks=b.max_m_blocks_per_core,
                nslice=b.nslice,
                wdepth=b.weight_depth,
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
    assert tt_weight.buffer_aligned_page_size() == ttnn.tile_size(spec["weight_dtype"])
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=spec["dtype"], layout=spec["layout"], device=device, memory_config=spec["mem"]
    )
    device.clear_program_cache()
    tt_out = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=spec["H"],
        input_width=spec["W"],
        bias=tt_bias,
        padding=spec["padding"],
        stride=spec["stride"],
        dilation=spec["dilation"],
        activation=spec["activation"],
        compute_kernel_config=spec["cfg"],
    )
    if _collecting(request):
        return  # the inline precompile pass (NO_DISPATCH): outputs are addr-0 garbage, log nothing
    rec = dict(case=case_token, variant=variant, layout=layout_id, blocking=blockings[-1] if blockings else None)
    rec["ablate"] = os.environ.get("CONV2D_3X3_ABLATE", "") if ABLATION else ""
    if not ABLATION:
        got = ttnn.to_torch(tt_out).float().reshape(-1, spec["c_out"])
        x = torch_input.float().reshape(1, spec["H"], spec["W"], spec["c_in"]).permute(0, 3, 1, 2)
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
        ref = ref.permute(0, 2, 3, 1).reshape(-1, spec["c_out"])
        _, pcc = comp_pcc(ref, got, spec["pcc"])
        rec["pcc"] = float(pcc)
        _outputs[(case_token, variant)] = got
        base = _outputs.get((case_token, base_variant))
        rec["equal_base"] = None if base is None else bool(torch.equal(base, got))
        # A later base compares back against the candidates (same walk) that ran before it.
        if variant == base_variant:
            rec["equal_earlier"] = {
                v: bool(torch.equal(o, got))
                for (c, v), o in _outputs.items()
                if c == case_token and v != variant and _parse_variant(v)[3] == base_variant
            }
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if not ABLATION:
        assert rec["pcc"] >= spec["pcc"], rec
        assert rec["equal_base"] is not False, f"{case_token} {variant}: not bit-identical to base"
        assert all(rec.get("equal_earlier", {}).values()), rec
