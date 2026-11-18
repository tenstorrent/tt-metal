# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: split the per-pixel row gather of conv2d_3x3 across both data-movement RISCs.

Variants (SRG_VARIANTS, comma-separated; one op dispatch per (case, variant), in that order):
  committed            the real op (ttnn.operations.conv2d_3x3), untouched
  off_dN               this dir's copy, split off, cb_im2col_rm depth N tile-rows
  split_pPwWaA_dN[_wp] copy, split on: writer owns tile-rows s with s % P >= P - W, grants A rows ahead, depth N;
                       _wp: the writer issues every free weight slot before its first rows
  dyn_qQ_dN[_wp]       copy, dynamic split: the reader hands the writer the next rows while it has < Q outstanding
  dyn2_qQ_dN           dynamic split v2: hand-outs across K-blocks; the writer issues weights whenever idle
  ..._nz               without the writer's per-K-block serve zone (profiler-only overhead)
  ..._ni / offni       the per-tile-row row gather compiled as a real call (noinline), split on / off

Cases (SRG_CASES, comma-separated): see CASES. Every dispatch is correctness-gated: PCC vs torch at the case's
threshold, and (when `committed` ran earlier in the session) bitwise equality with the committed output.

Run (profile):
  SRG_CASES=res4_0_conv2 SRG_VARIANTS=committed,off_d2,split_p2w1a1_d4 \\
    scripts/run_safe_pytest.sh --profile --run-all \\
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/split_row_gather/test_split_row_gather.py
then: python3 .../split_row_gather/report.py <ops_perf_results csv>
"""

import importlib
import json
import os
import re
import time
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
_real = importlib.import_module("ttnn.operations.conv2d_3x3")
_copy = importlib.import_module("ttnn.operations.conv2d_3x3.perf_experiments.split_row_gather.op")
_copy_pd = importlib.import_module(
    "ttnn.operations.conv2d_3x3.perf_experiments.split_row_gather.op.conv2d_3x3_program_descriptor"
)
_copy_op = importlib.import_module("ttnn.operations.conv2d_3x3.perf_experiments.split_row_gather.op.conv2d_3x3")

HS, BS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
LOFI = ttnn.MathFidelity.LoFi


def _grid(num_cores, grid_x=11, start=(0, 0)):
    full_rows, rem = divmod(num_cores, grid_x)
    x0, y0 = start
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + grid_x - 1, y0 + full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0 + full_rows), ttnn.CoreCoord(x0 + rem - 1, y0 + full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _height(shard, num_cores):
    return ttnn.MemoryConfig(
        HS, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(num_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    )


def _block(shard, gx, gy):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    return ttnn.MemoryConfig(BS, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


# name -> dict(shape=((H, W, C_in), C_out, padding, stride, dilation), dtype, layout, weight_dtype, mem (callable),
#              bias, relu, fidelity, pcc, force_bm)
CASES = {
    # Perf focus: Panoptic res4.0.conv2 @ LoFi (feature_spec LOOSE "[perf]" row): BLOCK 8x8, K split, stride 2.
    "res4_0_conv2": dict(
        shape=((64, 128, 256), 256, 1, 2, 1),
        dtype=ttnn.bfloat8_b,
        layout=TL,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _block([1024, 32], 8, 8),
        bias=True,
        relu=True,
        fidelity=LOFI,
        pcc=0.996,
    ),
    # HEIGHT TILE bf8_b, stride 2, weight multicast (writer = multicast sender/receiver), 86 cores.
    "res3_0_conv2": dict(
        shape=((128, 256, 128), 128, 1, 2, 1),
        dtype=ttnn.bfloat8_b,
        layout=TL,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _height([384, 128], 86),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.99,
    ),
    # DRAM interleaved TILE bf16, stride 2: the writer also stores the output.
    "dram_tile_s2": dict(
        shape=((64, 128, 128), 128, 1, 2, 1),
        dtype=ttnn.bfloat16,
        layout=TL,
        weight_dtype=ttnn.bfloat16,
        mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
        bias=True,
        relu=False,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.995,
    ),
    # DRAM interleaved RM bf16, C_in 48 (padded lanes: per-tap reads), stride 1.
    "dram_rm_c48": dict(
        shape=((64, 128, 48), 64, 1, 1, 1),
        dtype=ttnn.bfloat16,
        layout=RM,
        weight_dtype=ttnn.bfloat16,
        mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
        bias=True,
        relu=False,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.995,
    ),
    # HEIGHT RM bf16, stride 2, C_in 32: split slab fill (the writer also reads higher shards' slab runs).
    "height_rm_s2": dict(
        shape=((64, 128, 32), 64, 1, 2, 1),
        dtype=ttnn.bfloat16,
        layout=RM,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _height([128, 32], 64),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.99,
    ),
    # ---- correctness-only edges
    # HEIGHT TILE C_in 48 (lanes), valid padding, shards not on image rows.
    "height_tile_c48": dict(
        shape=((16, 34, 48), 64, 0, 1, 1),
        dtype=ttnn.bfloat16,
        layout=TL,
        weight_dtype=ttnn.bfloat16,
        mem=lambda: _height([160, 64], 4),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.997,
    ),
    # Several one-tile-row M-blocks per core, ragged last core outside the weight multicast, stride 2, RM.
    "height_rm_s2_bm1": dict(
        shape=((64, 128, 64), 64, 1, 2, 1),
        dtype=ttnn.bfloat16,
        layout=RM,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _height([768, 64], 11),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.99,
        force_bm=1,
    ),
    # BLOCK TILE: short last channel slice (16 of 32: its own reader AND writer kernel), short last N-slice.
    "block_short_slice": dict(
        shape=((16, 34, 48), 96, 0, 1, 1),
        dtype=ttnn.bfloat16,
        layout=TL,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _block([96, 32], 2, 6),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.997,
    ),
    # Untouched paths (the split never applies): tap-major (stride 1) and face-direct (stride * C_in == 16).
    "dram_tile_tapmajor": dict(
        shape=((32, 64, 128), 128, 1, 1, 1),
        dtype=ttnn.bfloat16,
        layout=TL,
        weight_dtype=ttnn.bfloat16,
        mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
        bias=True,
        relu=False,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.995,
    ),
    "height_rm_facedirect": dict(
        shape=((64, 128, 8), 64, 1, 2, 1),
        dtype=ttnn.bfloat16,
        layout=RM,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _height([1000, 8], 9),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.99,
    ),
    # BLOCK RM, stride 2, 2 M-blocks per core (forced), K split.
    "block_rm_s2_bm2": dict(
        shape=((32, 64, 64), 64, 1, 2, 1),
        dtype=ttnn.bfloat16,
        layout=RM,
        weight_dtype=ttnn.bfloat8_b,
        mem=lambda: _block([512, 32], 2, 4),
        bias=True,
        relu=True,
        fidelity=ttnn.MathFidelity.HiFi4,
        pcc=0.99,
        force_bm=2,
    ),
}

VARIANT_RE = re.compile(
    r"^(committed|off_d(\d+)|split_p(\d+)w(\d+)a(\d+)_d(\d+)(_wp)?|dyn(2?)_q(\d+)_d(\d+)(_wp)?(_nz)?|offni)(_ni)?$"
)


def _variant_knobs(name):
    m = VARIANT_RE.match(name)
    assert m, f"bad variant {name}"
    if name == "committed":
        return None
    noinline = dict(ROW_GATHER_NOINLINE=name.endswith("_ni") or name == "offni")
    if name == "offni":
        return dict(ROW_GATHER_SPLIT=False, **noinline)
    if name.startswith("off"):
        return dict(ROW_GATHER_SPLIT=False, IM2COL_RM_DEPTH_ROWS=int(m.group(2)), **noinline)
    if name.startswith("dyn"):
        return dict(
            **noinline,
            ROW_GATHER_SPLIT=True,
            ROW_DYNAMIC=2 if m.group(8) else 1,
            ROW_QUEUE=int(m.group(9)),
            ROW_SPLIT_DEPTH_ROWS=int(m.group(10)),
            ROW_SPLIT_WEIGHT_PREFETCH=m.group(11) is not None,
            ROW_SERVE_ZONE=m.group(12) is None,
        )
    p, w, a, d = (int(m.group(i)) for i in range(3, 7))
    return dict(
        **noinline,
        ROW_GATHER_SPLIT=True,
        ROW_SPLIT_PERIOD=p,
        ROW_SPLIT_WRITER_ROWS=w,
        ROW_GRANT_AHEAD=a,
        ROW_SPLIT_DEPTH_ROWS=d,
        ROW_SPLIT_WEIGHT_PREFETCH=m.group(7) is not None,
    )


SRG_CASES = [c for c in os.environ.get("SRG_CASES", "res4_0_conv2").split(",") if c]
SRG_VARIANTS = [v for v in os.environ.get("SRG_VARIANTS", "committed,off_d2,split_p2w1a1_d4").split(",") if v]
LOG = HERE / "runs" / f"run_{int(time.time())}_{os.getpid()}.jsonl"
# A precompile collect pass runs every test body once (without real device results) before the real pass, so the
# committed output is keyed by how many times this (case, variant) ran: only same-pass outputs are compared.
_committed_out = {}
_runs = {}


def _force_bm(pd_module, bm):
    select = pd_module.select_blocking

    def forced(g, grid, l1_budget):
        b = select(g, grid, l1_budget)
        fit = pd_module._footprint(g, b.core_m_tiles, bm, b.block_k_tiles, b.block_n_tiles)
        return pd_module.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=bm,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            **({} if "im2col_rm_depth_rows" in fit else dict(im2col_rm_depth_rows=b.im2col_rm_depth_rows)),
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    return forced


@pytest.mark.parametrize("variant", SRG_VARIANTS)
@pytest.mark.parametrize("case_name", SRG_CASES)
def test_split_row_gather(case_name, variant, device, monkeypatch):
    import torch

    case = CASES[case_name]
    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    memory_config = case["mem"]()
    knobs = _variant_knobs(variant)
    if knobs is None:
        pkg = _real
        op_module = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
        pd_module = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    else:
        pkg, op_module, pd_module = _copy, _copy_op, _copy_pd
        for k, v in knobs.items():
            monkeypatch.setattr(_copy_pd, k, v)
    pd_module.select_blocking.cache_clear()
    pd_module._block_extents.cache_clear()
    select = _force_bm(pd_module, case["force_bm"]) if case.get("force_bm") else pd_module.select_blocking
    blockings = []

    def recording_select(g, grid, l1_budget):
        b = select(g, grid, l1_budget)
        blockings.append(
            dict(
                bm=b.block_m_tiles,
                bk=b.block_k_tiles,
                nk=b.num_k_blocks,
                mblocks=b.max_m_blocks_per_core,
                depth=b.im2col_rm_depth_rows,
            )
        )
        return b

    monkeypatch.setattr(op_module, "select_blocking", recording_select)

    torch.manual_seed(0)
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias"] else None
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=memory_config,
        device=device,
    )
    tt_weight = pkg.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = pkg.prepare_conv2d_bias(torch_bias, **kwargs) if case["bias"] else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=memory_config
    )
    tt_output = pkg.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if case["relu"] else None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fidelity"], fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    out = ttnn.to_torch(tt_output).float().reshape(1, 1, -1, c_out)

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
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    passed, pcc = comp_pcc(ref, out, case["pcc"])
    rel_rms = float(torch.sqrt(torch.mean((out - ref) ** 2)) / torch.sqrt(torch.mean(ref**2)))
    pass_idx = _runs[(case_name, variant)] = _runs.get((case_name, variant), 0) + 1
    if variant == "committed":
        _committed_out[(case_name, pass_idx)] = out
    equal = None
    if (case_name, pass_idx) in _committed_out and variant != "committed":
        equal = bool(torch.equal(out, _committed_out[(case_name, pass_idx)]))
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(
            json.dumps(
                dict(
                    case=case_name,
                    variant=variant,
                    pcc=float(pcc),
                    rel_rms=rel_rms,
                    equal_committed=equal,
                    blocking=blockings[-1] if blockings else None,
                )
            )
            + "\n"
        )
    assert passed, f"{case_name}/{variant}: PCC {pcc} < {case['pcc']}"
    assert equal is not False, f"{case_name}/{variant}: output differs from the committed op"
