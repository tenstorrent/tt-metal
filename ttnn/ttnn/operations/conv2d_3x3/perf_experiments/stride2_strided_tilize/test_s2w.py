# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: stride-2 strided-window im2col for conv2d_3x3 (perf experiment, not part of the op).

Runs this dir's private copy of conv2d_3x3 (op_copy/, loaded by path; this dir has no __init__.py, so ttnn's
walk_packages never executes it). The variant is the S2W_VARIANT env var, which the copy's program descriptor turns
into reader / compute compile-time args:
  0 baseline (the committed op: per-pixel row gather split over reader + writer, width-3c tilize),
  1 strided window: one read per output-row run per kernel row, fast tilize at a 2-pixel row stride,
  2 strided window, per-tile tilize at a 2-pixel row stride.
  Further switches (op_copy only): S2W_DIRECT (1, default: slab-direct units on the N-slice walk), S2W_DEPTH (in-flight
  units of the pipelined reader, 0 default = one at a time), S2W_SPLIT (1: reader + writer hand-out), S2W_ABLATE.
  S2W_OP_DIR=graduate/op runs the clean integration candidate instead (saved as variant 9).

One process per variant, one dispatch per case in S2W_CASES order:

  S2W_VARIANT=1 S2W_CASES=res4_0_lofi S2W_TAG=v1a TT_METAL_PROFILER_DIR=/tmp/s2w/v1a \\
    scripts/run_safe_pytest.sh --profile --run-all \\
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/stride2_strided_tilize/test_s2w.py --import-mode=prepend

Every dispatch is correctness-gated (PCC vs torch at the case's threshold) and saves its output to S2W_OUT_DIR
(default /tmp/s2w_outputs) as <case>__v<variant>.pt for compare_outputs.py (torch.equal vs the baseline). Each dispatch
appends a JSON line (case, variant, pcc, gather path, blocking) to runs.jsonl; report.py joins it with the profiler
CSV's GenericOp rows.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
OPC = "stride2_strided_tilize_op_copy"


def _load_op_copy():
    if OPC not in sys.modules:
        # S2W_OP_DIR=graduate/op: the clean integration candidate (no switches; its outputs save as variant 9).
        root = HERE / os.environ.get("S2W_OP_DIR", "op_copy")
        spec = importlib.util.spec_from_file_location(
            OPC, root / "package_init.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC]


opc = _load_op_copy()
pd = sys.modules[OPC + ".conv2d_3x3_program_descriptor"]
op_mod = sys.modules[OPC + ".conv2d_3x3"]

VARIANT = getattr(pd, "S2W_VARIANT", 9)
if os.environ.get("S2W_KERNEL_DIR"):
    pd.KERNEL_DIR = Path(os.environ["S2W_KERNEL_DIR"])
OUT_DIR = Path(os.environ.get("S2W_OUT_DIR", "/tmp/s2w_outputs"))
RUNS = HERE / "runs.jsonl"

HS, BS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
LOFI, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4
DRAM = "dram"


def _row_major_cores(num_cores, grid_x):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _mem(spec, shape, layout, dtype, device):
    if spec == DRAM:
        return ttnn.DRAM_MEMORY_CONFIG
    if spec in ("B_derived", "H_derived"):
        from eval.golden_tests.conv2d_3x3 import helpers

        ml = BS if spec == "B_derived" else HS
        return helpers.build_input_memory_config(shape, ml, layout=layout, dtype=dtype, device=device, extras=None)
    kind, shard, grid = spec
    if kind == "H":
        cores = _row_major_cores(grid, 11)
        return ttnn.MemoryConfig(HS, ttnn.BufferType.L1, ttnn.ShardSpec(cores, shard, ttnn.ShardOrientation.ROW_MAJOR))
    gx, gy = grid
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    return ttnn.MemoryConfig(BS, ttnn.BufferType.L1, ttnn.ShardSpec(cores, shard, ttnn.ShardOrientation.ROW_MAJOR))


def _case(shape, dtype, layout, wdtype, mem, *, bias=True, relu=False, fidelity=None, pcc=0.99):
    return dict(
        shape=shape, dtype=dtype, layout=layout, wdtype=wdtype, mem=mem, bias=bias, relu=relu, fid=fidelity, pcc=pcc
    )


# name -> case; shape = ((H, W, C_in), C_out, padding, stride, dilation)
CASES = {
    # Focus: feature_spec LOOSE "res4.0.conv2 @ LoFi [perf]" (BLOCK 8x8 [1024, 32], bf8_b x bf8_b, bias + ReLU, LoFi,
    # fp32 DEST), with the focus harness's input generators.
    "res4_0_lofi": _case(
        ((64, 128, 256), 256, 1, 2, 1),
        ttnn.bfloat8_b,
        TL,
        ttnn.bfloat8_b,
        ("B", [1024, 32], (8, 8)),
        relu=True,
        fidelity=LOFI,
        pcc=0.996,
    ),
    # BLOCK harness res4_0_conv2 (HiFi4).
    "res4_0_hifi4": _case(
        ((64, 128, 256), 256, 1, 2, 1),
        ttnn.bfloat8_b,
        TL,
        ttnn.bfloat8_b,
        ("B", [1024, 32], (8, 8)),
        relu=True,
        fidelity=HIFI4,
    ),
    # HEIGHT harness res3_0_conv2: stride 2, C 128, 86 cores, default compute config.
    "res3_0": _case(
        ((128, 256, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, ("H", [384, 128], 86), relu=True
    ),
    # res3.0 at the model's LoFi config.
    "res3_0_lofi": _case(
        ((128, 256, 128), 128, 1, 2, 1),
        ttnn.bfloat8_b,
        TL,
        ttnn.bfloat8_b,
        ("H", [384, 128], 86),
        relu=True,
        fidelity=LOFI,
        pcc=0.996,
    ),
    # Interleaved stride 2 (C_in 128: 4 tiles per pixel -> fast-tilize chunks 4 + 4 + 4).
    "dram_s2_tile_bf16": _case(((64, 128, 128), 128, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "dram_s2_tile_bf8b": _case(((64, 128, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, DRAM),
    "dram_s2_rm_bf16": _case(((64, 128, 128), 128, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    # feature_spec INPUTS stride 2: W_out 32 (window) and W_out 16 with even W (fallback: shared slot not zero).
    "golden_s2_c128_tile": _case(((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat16, DRAM),
    "golden_s2_c128_rm": _case(((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    "golden_s2_c64_w16_tile": _case(((32, 32, 64), 128, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    # W_out 31 with odd W: tile-rows straddle output rows and share the zero slot (C_in 64: chunks 4 + 2).
    "s2_w61_tile": _case(((64, 61, 64), 64, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "s2_w61_rm": _case(((64, 61, 64), 64, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    # C_in 32 interleaved (chunk 3), right edge at odd W with W_out 32.
    "s2_w63_c32_tile": _case(((32, 63, 32), 64, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    # C_in 96: 3 tiles per pixel, a 9-tile kernel row -> fast-tilize chunks 4 + 2 + 3.
    "s2_c96_tile": _case(((32, 64, 96), 64, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    # BLOCK stride 2 at derived geometry (golden grid), and a short last channel slice (C_in 96).
    "golden_s2_block": _case(((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, "B_derived"),
    "s2_block_c96": _case(((32, 64, 96), 64, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, "B_derived"),
    "golden_s2_height": _case(((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, "H_derived"),
    # Short last channel slice with zeroed lanes (C_in 96 over 64-wide slices: 64 + 32 real + 32 zero lanes).
    "s2_block_short_slice": _case(
        ((32, 64, 96), 64, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, ("B", [256, 64], (2, 8))
    ),
    # BLOCK N-slice walk with several M-blocks per core (core_m_tiles 32): the slab outlives each block's direct tilize.
    "s2_block_big": _case(((128, 256, 64), 64, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, ("B", [4096, 32], (2, 8))),
    "s2_block_multi_m": _case(
        ((128, 256, 64), 64, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, ("B", [8192, 32], (2, 4))
    ),
    # ROW_MAJOR sharded stride 2: BLOCK (channel-slice pixel reads) and HEIGHT (split slab fill).
    "s2_block_rm": _case(((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, "B_derived", pcc=0.995),
    "s2_block_rm32": _case(
        ((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, ("B", [256, 32], (4, 8)), pcc=0.995
    ),
    "s2_height_rm": _case(((64, 64, 64), 64, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, "H_derived", pcc=0.995),
    # Dilated + strided (excluded: falls back to the row gather) and padding 0 (excluded at W_out 31).
    "golden_dil2_s2_tile": _case(((32, 32, 64), 64, 2, 2, 2), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "valid_s2_w31_rm": _case(((65, 64, 64), 64, 0, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
}

SELECTED = [c for c in os.environ.get("S2W_CASES", "res4_0_lofi").split(",") if c]
if SELECTED == ["all"]:
    SELECTED = list(CASES)


def _capture_descriptor():
    """Wrap the copy's create_program_descriptor to keep the last descriptor (gather path, blocking)."""
    seen = {}
    orig = op_mod.create_program_descriptor

    def wrapped(*args, **kwargs):
        d = orig(*args, **kwargs)
        seen["desc"] = d
        return d

    op_mod.create_program_descriptor = wrapped
    return seen, orig


def _gather_info(desc):
    ct = list(desc.kernels[0].compile_time_args)
    c_in, c_in_p, stride, dil = ct[7], ct[8], ct[10], ct[11]
    info = dict(
        c_in=c_in, c_in_p=c_in_p, stride=stride, dilation=dil, bm=ct[12], bk=ct[13], kkh=ct[14], k_blocks=ct[15]
    )
    tap_major, face_direct, row_split, window = ct[34], ct[35], ct[51], ct[61]
    info["gather"] = "tap_major" if tap_major else ("face_direct" if face_direct else ("window" if window else "row"))
    info["row_split"] = bool(row_split)
    info["nslice"] = bool(ct[47])
    return info


@pytest.mark.parametrize("name", SELECTED)
def test_s2w(name, device):
    import torch

    case = CASES[name]
    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    torch.manual_seed(0)
    memory_config = _mem(case["mem"], (1, 1, H * W, c_in), case["layout"], case["dtype"], device)
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["wdtype"],
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias"] else None
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if case["bias"] else None
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=memory_config
    )
    extra = {}
    if case["fid"] is not None:
        extra["compute_kernel_config"] = ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fid"], fp32_dest_acc_en=True, math_approx_mode=False
        )
    seen, orig = _capture_descriptor()
    try:
        tt_output = opc.conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation="relu" if case["relu"] else None,
            **extra,
        )
    finally:
        op_mod.create_program_descriptor = orig
    out = ttnn.to_torch(tt_output).float()

    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if case["relu"]:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(-1, c_out)
    out = out.reshape(-1, out.shape[-1])[: ref.shape[0], :c_out]
    pcc = float(torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_DIR / f"{name}__v{VARIANT}.pt")
    rec = dict(
        t=time.time(), tag=os.environ.get("S2W_TAG", ""), case=name, variant=VARIANT, pcc=pcc, ok=pcc >= case["pcc"]
    )
    if "desc" in seen:
        rec.update(_gather_info(seen["desc"]))
    with open(RUNS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert pcc >= case["pcc"], f"{name} v{VARIANT}: pcc {pcc:.6f} < {case['pcc']}"
