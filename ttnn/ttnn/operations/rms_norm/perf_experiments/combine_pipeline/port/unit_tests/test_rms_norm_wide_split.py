# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5 — interleaved W split past 8 Tensix cores (owner combine, groups spanning grid rows).

Pins:
  * the per-core-cost split choosing k > 8 (k > grid_x on the 13x10 BH p150b grid) for few-tile-row wide rows;
  * correctness of groups placed row-major across grid rows (forced k), with the owner combine, the per-core gamma
    load finished after the partial send (GAMMA_AFTER_SEND), and the flush signal fence (SIGNAL_AFTER_FLUSH) —
    across TILE / ROW_MAJOR input, W / H tails, TILE / ROW_MAJOR gamma, bf16 / fp32 / bf8b gamma, fp32 DEST,
    multi-block owner exchange (OVERLAP_BLOCKS = 2), and both knobs off.
"""

import pytest
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    import torch  # function-local: the ttnn tree forbids global torch imports

    xf = x.to(torch.float32)
    y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        y = y * gamma.to(torch.float32).reshape(-1)
    return y


def pcc(actual, expected):
    import torch  # function-local: the ttnn tree forbids global torch imports

    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def perf_config():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def blocking(device, shape, layout, x_dtype=ttnn.bfloat16, fp32_dest=False):
    geo = pd.derive_geometry(shape)
    sz = pd._Sizes(x_dtype, ttnn.bfloat16, layout == ttnn.ROW_MAJOR_LAYOUT, True, fp32_dest)
    grid = device.compute_with_storage_grid_size()
    elem = 4 if x_dtype == ttnn.float32 else 2
    return pd.select_blocking(geo, sz, grid.x, grid.y, pd.cb_l1_budget(device), elem), grid


def run(
    device,
    shape,
    *,
    layout=ttnn.TILE_LAYOUT,
    x_dtype=ttnn.bfloat16,
    gamma_dtype=ttnn.bfloat16,
    gamma_layout=ttnn.TILE_LAYOUT,
    use_gamma=True,
    cfg=None,
    pcc_min=0.9995,
):
    import torch  # function-local: the ttnn tree forbids global torch imports

    torch.manual_seed(5)
    x = torch.randn(shape, dtype=torch.float32)
    if x_dtype == ttnn.bfloat16:
        x = x.to(torch.bfloat16)
    W = shape[-1]
    tt_x = ttnn.from_torch(x, dtype=x_dtype, layout=layout, device=device)
    tt_g, g_ref = None, None
    if use_gamma:
        g = torch.randn(1, 1, 1, W, dtype=torch.float32)
        tt_g = ttnn.from_torch(g, dtype=gamma_dtype, layout=gamma_layout, device=device)
        g_ref = ttnn.to_torch(tt_g).to(torch.float32).reshape(-1)[:W]
    out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=cfg or perf_config())
    actual = ttnn.to_torch(out).to(torch.float32)
    assert list(actual.shape) == list(shape)
    p = pcc(actual, torch_rms_norm(x, g_ref))
    assert p >= pcc_min, f"PCC {p:.6f} < {pcc_min} shape={shape} layout={layout}"


def force_k(monkeypatch, k):
    monkeypatch.setattr(pd, "_choose_k", lambda *a, **kw: k)


# The cost model's own choice: k > 8 on few-tile-row wide rows (k > grid_x on the 13x10 grid -> spanning groups).
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 32768), (1, 1, 32, 7168), (1, 1, 160, 11008), (1, 1, 96, 6144)], ids=lambda s: f"{s[2]}x{s[3]}"
)
def test_wide_split_chosen(device, shape, layout):
    blk, grid = blocking(device, shape, layout)
    assert blk.k > pd.K_SYM_MAX_INTERLEAVED and blk.opts.owner_combine, (shape, blk.k)
    if (grid.x, grid.y) == (13, 10) and layout == ttnn.TILE_LAYOUT:
        assert blk.k > grid.x, (shape, blk.k)
    run(device, shape, layout=layout)


# Forced k spanning grid rows: tails, gamma formats, no gamma, fp32 input / DEST.
@pytest.mark.parametrize("k", [12, 20, 40])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_forced_k_non_aligned(device, monkeypatch, k, layout):
    force_k(monkeypatch, k)
    run(device, (2, 1, 100, 10000), layout=layout, gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize(
    "gamma_dtype, gamma_layout",
    [(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), (ttnn.float32, ttnn.TILE_LAYOUT), (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)],
    ids=["g_bf16_rm", "g_fp32_tile", "g_bf8b_tile"],
)
def test_forced_k_gamma_formats(device, monkeypatch, gamma_dtype, gamma_layout):
    force_k(monkeypatch, 30)
    run(device, (1, 1, 64, 8192), gamma_dtype=gamma_dtype, gamma_layout=gamma_layout, pcc_min=0.999)


def test_forced_k_no_gamma(device, monkeypatch):
    force_k(monkeypatch, 26)
    run(device, (1, 1, 96, 6144), use_gamma=False)


def test_forced_k_fp32(device, monkeypatch):
    force_k(monkeypatch, 24)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    run(device, (1, 1, 64, 6144), x_dtype=ttnn.float32, gamma_dtype=ttnn.float32, cfg=cfg, pcc_min=0.9999)


# Multi-block owner exchange across spanning groups: several blocks per core (OVERLAP_BLOCKS = 2 -> depth 2,
# x / x_alt alternate), gamma finished after block 0's send only. 5 groups of 26 (spanning grid rows) on 13x10:
#   30 tile-rows -> 6 per group, blocks 3 + 3 (unicast inv_rms return only)
#   10 tile-rows -> 2 per group, blocks 1 + 1 (Perf 1 single-owner multicast return, flag re-armed per block)
#   15 tile-rows -> 3 per group, blocks 2 + 1 (unicast block, then a multicast block on the same semaphores)
@pytest.mark.parametrize("tile_rows", [30, 10, 15])
def test_forced_k_multi_block(device, monkeypatch, tile_rows):
    force_k(monkeypatch, 26)
    monkeypatch.setattr(pd, "OVERLAP_BLOCKS", 2)
    shape = (1, 1, 32 * tile_rows, 4096)
    blk, _ = blocking(device, shape, ttnn.TILE_LAYOUT)
    rows = blk.groups[0][0].core_rows
    assert pd._div_up(rows, blk.block_row_tiles) > 1, (rows, blk.block_row_tiles)
    run(device, shape)


# Both new knobs at their trivial (Refinement 4) setting stay correct.
def test_knobs_off(device, monkeypatch):
    monkeypatch.setattr(pd, "GAMMA_AFTER_SEND", False)
    monkeypatch.setattr(pd, "SIGNAL_AFTER_FLUSH", False)
    run(device, (1, 1, 32, 16384))
    run(device, (1, 1, 352, 4096), layout=ttnn.ROW_MAJOR_LAYOUT)
