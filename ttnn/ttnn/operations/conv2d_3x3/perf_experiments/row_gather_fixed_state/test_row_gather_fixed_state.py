# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: fixed-state row gather for conv2d_3x3 (perf experiment, not part of the op).

Runs this dir's private copy of conv2d_3x3 (op_copy/, loaded by path: this tree has no __init__.py, so ttnn's
walk_packages never executes it). The row gather body is picked by the RG_VARIANT env var, which the copy's program
descriptor turns into a reader / writer define (see op_copy/kernels/conv2d_3x3_gather_dataflow.hpp):
  0 baseline (the committed per-pixel row gather), 1 fixed-state runs, 2 + out-of-image run zero fill,
  3 2 + cold generic body, 4 2 + tile_row inlined, 5 2 + per-run trid throttle count.

One process per variant (the variant is a kernel define), one dispatch per (case) in RGFS_CASES order:

  RG_VARIANT=2 RGFS_CASES=res4_0_lofi TT_METAL_PROFILER_DIR=/tmp/rgfs/v2 \\
    scripts/run_safe_pytest.sh --profile --run-all \\
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/row_gather_fixed_state/test_row_gather_fixed_state.py

Every dispatch is correctness-gated (PCC vs torch at the case's threshold) and saves its output to RGFS_OUT_DIR
(default /tmp/rgfs_outputs) as <case>__v<variant>.pt for compare_outputs.py (bitwise vs the baseline). Each dispatch
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
OPC = "row_gather_fixed_state_op_copy"


def _load_op_copy():
    if OPC not in sys.modules:
        root = HERE / "op_copy"
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

VARIANT = int(os.environ.get("RG_VARIANT", "0"))
# RGFS_GRADUATE=1: the committed kernels with graduate/kernels/conv2d_3x3_gather_dataflow.hpp (the patch as it would
# land in the op, no RG_VARIANT switch); run it with RG_VARIANT=9 so its outputs are saved under their own name.
if os.environ.get("RGFS_GRADUATE") == "1":
    pd.KERNEL_DIR = HERE / "graduate" / "kernels"
OUT_DIR = Path(os.environ.get("RGFS_OUT_DIR", "/tmp/rgfs_outputs"))
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


def _mem(spec):
    if spec == DRAM:
        return ttnn.DRAM_MEMORY_CONFIG
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
    # Focus: feature_spec LOOSE "res4.0.conv2 @ LoFi [perf]" (BLOCK 8x8 [1024, 32], bf8_b, bias + ReLU, LoFi, fp32 DEST).
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
    # HEIGHT harness res3_0_conv2: stride 2, C 128, 86 cores.
    "res3_0": _case(
        ((128, 256, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat8_b, ("H", [384, 128], 86), relu=True
    ),
    # Interleaved stride 2 (row gather, fixed path).
    "dram_s2_tile_bf16": _case(((64, 128, 128), 128, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "dram_s2_rm_bf16": _case(((64, 128, 128), 128, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    # HEIGHT ROW_MAJOR C_in 48 (shard width 48): unpadded lanes, K-padded kernel row -> fixed path, tail read size.
    "height_rm_c48": _case(
        ((32, 64, 48), 64, 1, 1, 1), ttnn.bfloat16, RM, ttnn.bfloat16, ("H", [128, 48], 16), pcc=0.995
    ),
    "height_rm_c48_s2": _case(
        ((64, 64, 48), 64, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, ("H", [256, 48], 16), pcc=0.995
    ),
    # feature_spec INPUTS, correctness of the generic path: dilated + strided, C_in not a multiple of 32.
    "golden_dil2_s2_rm": _case(((32, 32, 64), 64, 2, 2, 2), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    "golden_dil2_s2_tile": _case(((32, 32, 64), 64, 2, 2, 2), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "golden_c48_tile": _case(((14, 16, 48), 64, 1, 1, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "golden_c48_rm": _case(((14, 16, 48), 64, 1, 1, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    "golden_c8_s2_rm": _case(((32, 32, 8), 32, 1, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    "golden_s2_tile_bf8b": _case(((32, 64, 128), 128, 1, 2, 1), ttnn.bfloat8_b, TL, ttnn.bfloat16, DRAM),
    # Correctness of the fixed window's edges (the op supports padding 0 / d and 32-aligned H*W, H_out*W_out only):
    # padding 0, odd output widths (tile-rows straddle output rows: several runs per tile-row), a stride-2 right edge.
    "valid_s2_w31_rm": _case(((65, 64, 64), 64, 0, 2, 1), ttnn.bfloat16, RM, ttnn.bfloat16, DRAM, pcc=0.995),
    "s2_w31_tile": _case(((64, 62, 64), 64, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
    "s2_right_edge_tile": _case(((32, 63, 64), 64, 1, 2, 1), ttnn.bfloat16, TL, ttnn.bfloat16, DRAM, pcc=0.995),
}
# The interleaved perf harness (test_conv2d_3x3_perf): 6 shapes x 3 layouts, stride 1, bias, bf16 weights.
for (H, W, ci), co, p in [
    ((32, 64, 128), 128, 1),
    ((16, 16, 64), 128, 1),
    ((8, 16, 512), 256, 1),
    ((8, 16, 2048), 256, 1),
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
]:
    for lid, dt, lay in [
        ("rm_bf16", ttnn.bfloat16, RM),
        ("tile_bf16", ttnn.bfloat16, TL),
        ("tile_bf8b", ttnn.bfloat8_b, TL),
    ]:
        CASES[f"il_H{H}_W{W}_Ci{ci}_{lid}"] = _case(
            ((H, W, ci), co, p, 1, 1), dt, lay, ttnn.bfloat16, DRAM, pcc=0.995 if dt == ttnn.bfloat16 else 0.99
        )

SELECTED = [c for c in os.environ.get("RGFS_CASES", "res4_0_lofi").split(",") if c]
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
    tap_major, face_direct, row_split = ct[34], ct[35], ct[51]
    info["gather"] = "tap_major" if tap_major else ("face_direct" if face_direct else "row")
    info["row_split"] = bool(row_split)
    info["fixed_capable"] = info["gather"] == "row" and c_in == c_in_p and dil == 1
    return info


@pytest.mark.parametrize("name", SELECTED)
def test_row_gather_fixed_state(name, device):
    import torch

    case = CASES[name]
    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    torch.manual_seed(0)
    memory_config = _mem(case["mem"])
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
        t=time.time(), tag=os.environ.get("RGFS_TAG", ""), case=name, variant=VARIANT, pcc=pcc, ok=pcc >= case["pcc"]
    )
    if "desc" in seen:
        rec.update(_gather_info(seen["desc"]))
    with open(RUNS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert pcc >= case["pcc"], f"{name} v{VARIANT}: pcc {pcc:.6f} < {case['pcc']}"
