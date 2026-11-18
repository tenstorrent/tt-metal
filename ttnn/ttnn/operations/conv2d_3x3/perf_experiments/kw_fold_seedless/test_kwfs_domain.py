# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness domain of the seedless kw-fold copy beyond the [perf] cases: kw-fold forced on (_kw_fold_pays patched
to True in both the base op and the copy, so both run kw-fold) over dtypes (the bf16 bias decode), dilation, C_out
off the tile grid, several M-blocks per core, K-blocked and whole-K, HiFi4. Checks both against a fp32 torch conv and
against each other.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_seedless/test_kwfs_domain.py
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwfs_copy import copy_modules  # noqa: E402

OUT = Path(__file__).resolve().parent / "domain.jsonl"

# (H, W, C_in, C_out, d, dtype, weight_dtype, fidelity, relu, num_cores)
CASES = [
    ("n1_bf8", 32, 64, 64, 32, 1, "bf8", "bf8", "LoFi", True, 16),
    ("bf16_bias_kblk", 32, 64, 128, 128, 1, "bf16", "bf16", "LoFi", True, 16),
    ("bf16_act_bf8_w", 32, 64, 64, 64, 1, "bf16", "bf8", "HiFi4", False, 16),
    ("d2_bf8", 64, 64, 64, 64, 2, "bf8", "bf8", "LoFi", True, 32),
    ("cout48_bf8", 32, 96, 64, 48, 1, "bf8", "bf8", "LoFi", True, 24),
    ("multiblock_bf8", 64, 128, 64, 64, 1, "bf8", "bf8", "LoFi", True, 8),
    ("kblk_multiblock_bf8", 64, 128, 128, 128, 1, "bf8", "bf8", "HiFi4", True, 8),
    ("ragged_bf8", 48, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 20),
    ("many_blocks_wholek_bf8", 128, 256, 32, 32, 1, "bf8", "bf8", "LoFi", True, 4),
    ("many_blocks_kblk_bf8", 64, 128, 128, 128, 1, "bf8", "bf8", "LoFi", True, 8),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_kwfs_domain(case, device):
    import torch
    import ttnn

    tag, H, W, c_in, c_out, d, act, wdt, fid, relu, cores = case
    dt = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    dtype, weight_dtype = dt[act], dt[wdt]
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    shard_grid = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [per_core * 32, c_in], ttnn.ShardOrientation.ROW_MAJOR),
    )
    base_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    base_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    copy_op, copy_pd = copy_modules()
    torch.manual_seed(0)
    tx = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tw = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    tb = torch.randn(c_out).to(torch.bfloat16)
    kwargs = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=True, packer_l1_acc=False
    )
    xq = ttnn.to_torch(ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    wq = ttnn.to_torch(ttnn.from_torch(tw, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()
    bq = ttnn.to_torch(ttnn.from_torch(tb.reshape(1, c_out), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()[0]
    ref = torch.nn.functional.conv2d(xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2), wq, bias=bq, padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(-1, c_out)
    outs, taken = {}, {}
    for name, op_mod, pd in (("base", base_op, base_pd), ("copy", copy_op, copy_pd)):
        pays, mode, blk = pd._kw_fold_pays, pd.kw_fold_mode, pd._kw_fold_blocking
        seen = []
        pd._kw_fold_pays = lambda *a: True
        pd.kw_fold_mode = lambda *a, _m=mode, _s=seen: _s.append(_m(*a)) or _s[-1]
        pd._kw_fold_blocking = (
            lambda *a, _b=blk, _s=seen: _s.append(
                {
                    "bm": (r := _b(*a)).block_m_tiles,
                    "blocks": r.max_m_blocks_per_core,
                    "kblocked": pd.kw_fold_kblocked(a[0]),
                }
            )
            or r
        )
        try:
            x = ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            y = op_mod.conv2d_3x3(
                x,
                op_mod.prepare_conv2d_weights(tw, **kwargs),
                input_height=H,
                input_width=W,
                bias=op_mod.prepare_conv2d_bias(tb, **kwargs),
                padding=d,
                stride=1,
                dilation=d,
                activation="relu" if relu else None,
                compute_kernel_config=cfg,
            )
            outs[name] = ttnn.to_torch(y).float().reshape(-1, c_out)
        finally:
            pd._kw_fold_pays, pd.kw_fold_mode, pd._kw_fold_blocking = pays, mode, blk
        taken[name] = seen
    rec = {"case": tag, "kw_fold": taken}
    for name, o in outs.items():
        a, b = o.flatten(), ref.flatten()
        rec[name] = {
            "pcc": round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 6),
            "mae": round((a - b).abs().mean().item(), 5),
        }
    diff = outs["copy"] - outs["base"]
    rec["identical_frac"] = round((diff == 0).float().mean().item(), 6)
    rec["max_base_copy_diff"] = round(diff.abs().max().item(), 5)
    print("KWFS_DOM", json.dumps(rec))
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert True in taken["copy"], f"kw-fold not taken: {taken}"
    assert rec["copy"]["pcc"] >= 0.996 and rec["copy"]["pcc"] >= rec["base"]["pcc"] - 2e-4, rec
