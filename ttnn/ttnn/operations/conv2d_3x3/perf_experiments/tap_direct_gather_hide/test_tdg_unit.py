# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the op copy's candidate (TDG_CAND) on the op's own BLOCK unit tests (perf experiment).

Re-binds tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py to this dir's op copy and runs
its tests with the candidate knobs, plus multi-M-block cases at a grouped ring. Run with --import-mode=prepend.
"""

import importlib
import os

import pytest

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.tap_direct_gather_hide.op_copy"
_op = importlib.import_module(_PKG + ".conv2d_3x3")
_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
_ut = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded")

CAND = dict(TDG_LEAN=True, TDG_ZERO_ONCE=True, TDG_FIXED=True, TDG_GROUP=3, TDG_RING=6)
if os.environ.get("TDG_UNIT_BASELINE"):
    CAND = {}

for _name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
    setattr(_ut, _name, getattr(_op, _name))
_ut._op_module = _op
_ut._pd = _pd


@pytest.fixture(autouse=True)
def _candidate(monkeypatch):
    for k, v in CAND.items():
        monkeypatch.setattr(_pd, k, v)
    _pd.select_blocking.cache_clear()
    yield
    _pd.select_blocking.cache_clear()


test_conv2d_3x3_block_sharded = _ut.test_conv2d_3x3_block_sharded
test_conv2d_3x3_block_sharded_relu_after_combine = _ut.test_conv2d_3x3_block_sharded_relu_after_combine
test_conv2d_3x3_block_sharded_col_major_offset_grid = _ut.test_conv2d_3x3_block_sharded_col_major_offset_grid
test_conv2d_3x3_block_sharded_multi_m_block = _ut.test_conv2d_3x3_block_sharded_multi_m_block
test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_row = (
    _ut.test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_row
)
test_conv2d_3x3_block_sharded_writer_only_scatter = _ut.test_conv2d_3x3_block_sharded_writer_only_scatter


@pytest.mark.parametrize("bm", [1, 2], ids=["bm1", "bm2"])
@pytest.mark.parametrize("ring", [4, 6], ids=["ring4", "ring6"])
@pytest.mark.parametrize("dtype,layout", [_ut.LAYOUTS[1], _ut.LAYOUTS[2]])
@pytest.mark.parametrize("pd_case", [(1, 1), (2, 2), (4, 4)], ids=["p1d1", "p2d2", "p4d4"])
def test_tdg_multi_m_block_grouped(pd_case, dtype, layout, ring, bm, device, monkeypatch):
    """Several M-blocks per core at a grouped ring: the zero-once mask and the groups cross M-blocks (groups are cut at
    the ring end, so a short last group shifts the next block's slots)."""
    import torch

    torch.manual_seed(7)
    p, d = pd_case
    (H, W, c_in), c_out = (16, 64, 64), 96
    shard = (128, 32)
    config = _ut._block_config(H * W, c_in, layout, shard)
    real_select = _pd.select_blocking

    def small_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.k_split and b.core_m_tiles == 4
        fit = _pd._footprint(
            g, b.core_m_tiles, bm, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice, im2col_depth=ring
        )
        assert _pd._tap_direct(g, bm, b.block_k_tiles, b.nslice)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=bm,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(_op, "select_blocking", small_blocks)
    _ut._run(((H, W, c_in), c_out, p, 1, d), dtype, layout, config, device, with_bias=True, relu=True)


# Bit-exact: candidate vs the copy's default (= committed op) on the same inputs, one process.
# ((H, W, C_in), C_out, p, d, shard (rows, width)), forced bm (None = solver), forced ring (None = solver)
BITEXACT = [
    (((32, 64, 512), 512, 4, 4, (224, 64)), None, None),  # res5.1 geometry (W_out % 32 == 0, zero-once)
    (((32, 64, 512), 512, 8, 8, (224, 64)), None, None),  # res5.2
    (((32, 64, 512), 512, 2, 2, (224, 64)), None, None),  # res5.0
    (((16, 64, 64), 96, 1, 1, (128, 32)), 1, 6),  # multi-M-block, grouped ring
    (((16, 64, 64), 96, 4, 4, (128, 32)), 2, 4),
    (((32, 64, 64), 32, 8, 8, (256, 32)), None, None),  # d 8 halo 8 shards away
    (((16, 16, 64), 128, 1, 1, (64, 32)), None, None),  # W_out 16: no zero-once, general gathers
    (((16, 48, 64), 64, 1, 1, (96, 32)), None, None),  # W_out 48: tile-rows straddle output rows
    (((16, 96, 64), 64, 2, 2, (192, 32)), None, None),  # W_out 96: three column classes
    (((32, 32, 64), 64, 4, 4, (128, 32)), None, None),  # W_out 32: both edges in one unit
    (((16, 32, 64), 64, 36, 36, (128, 32)), None, None),  # d > 32: whole-zero taps
    (((16, 34, 48), 96, 0, 1, (96, 32)), None, None),  # short last channel slice, no padding
    (((8, 16, 256), 256, 1, 1, (32, 32)), None, None),  # res4-like, one tap tile wide
    (((16, 64, 160), 64, 2, 2, (256, 96)), None, None),  # short last slice (64 of 96) on tap-direct
    (((16, 64, 160), 64, 18, 18, (256, 96)), None, None),  # aspp3-like short slice, d 18
    (((32, 64, 128), 64, 12, 12, (256, 32)), None, None),  # aspp-like dilation, W_out 64
]


@pytest.mark.parametrize(
    "case,bm,ring", BITEXACT, ids=[f"H{c[0][0]}W{c[0][1]}C{c[0][2]}p{c[2]}d{c[3]}bm{b}r{r}" for c, b, r in BITEXACT]
)
def test_tdg_bit_exact(case, bm, ring, device, monkeypatch):
    import torch
    import ttnn

    (H, W, c_in), c_out, p, d, shard = case
    torch.manual_seed(11)
    layout = ttnn.TILE_LAYOUT
    config = _ut._block_config(H * W, c_in, layout, shard)
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    kwargs = dict(
        stride=1,
        padding=p,
        dilation=d,
        input_dtype=ttnn.bfloat8_b,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=config,
        device=device,
    )
    real_select = _pd.select_blocking
    used = {}

    def forced(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        used["td"] = _pd._tap_direct(g, b.block_m_tiles, b.block_k_tiles, b.nslice)
        if bm is None and ring is None:
            used["ring"] = b.im2col_rm_depth_rows
            return b
        m = bm or b.block_m_tiles
        fit = _pd._footprint(g, b.core_m_tiles, m, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice, im2col_depth=ring)
        used["ring"] = fit["im2col_rm_depth_rows"]
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=m,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(_op, "select_blocking", forced)
    outs = []
    for knobs in (
        {k: False if isinstance(v, bool) else (1 if k == "TDG_GROUP" else None) for k, v in CAND.items()},
        CAND,
    ):
        for k, v in knobs.items():
            monkeypatch.setattr(_pd, k, v)
        _pd.select_blocking.cache_clear()
        tt_weight = _op.prepare_conv2d_weights(torch_weight, **kwargs)
        tt_bias = _op.prepare_conv2d_bias(torch_bias, **kwargs)
        tt_input = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat8_b, layout=layout, device=device, memory_config=config
        )
        out = _op.conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=p,
            stride=1,
            dilation=d,
            activation="relu",
        )
        outs.append((ttnn.to_torch(out).float(), dict(used)))
    (base, bu), (cand, cu) = outs
    ref = _ut._reference(torch_input, torch_weight, torch_bias, H, W, p, 1, d, True)
    from tests.ttnn.utils_for_testing import comp_pcc

    ok, pcc = comp_pcc(ref, cand, 0.99)
    print(f"TDG_BITEXACT {case} bm={bm} ring base={bu} cand={cu} equal={torch.equal(base, cand)} pcc={pcc}")
    assert ok, pcc
    assert torch.equal(base, cand)
