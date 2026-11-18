# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the op copy's candidate (NCT_UNIT_MODE, default 1) on the op's own BLOCK unit tests (perf experiment).

Re-binds tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_block_sharded.py to this dir's op copy and runs its
tests with NSLICE_CENTER set, plus dilated / short-slice / multi-M-block N-slice cases where the center path applies.
Run with --import-mode=prepend.
"""

import importlib
import os

import pytest

# NCT_UNIT_PKG=op_grad: the graduation candidate (NSLICE_CENTER_TAPS on).
_SUB = os.environ.get("NCT_UNIT_PKG", "op_copy")
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_center_taps." + _SUB
_op = importlib.import_module(_PKG + ".conv2d_3x3")
_pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
_ut = importlib.import_module("tests.ttnn.unit_tests.operations.conv2d_3x3.test_conv2d_3x3_block_sharded")

MODE = int(os.environ.get("NCT_UNIT_MODE", "1"))
# Extra knobs, e.g. NCT_UNIT_KNOBS="NCT_COMPACT=1,NCT_GROUP=6,NCT_RING=12"
KNOBS = {
    k: (None if v == "None" else int(v))
    for k, v in (kv.split("=") for kv in os.environ.get("NCT_UNIT_KNOBS", "").split(",") if kv)
}


def _mode(g, b):
    if _SUB == "op_grad":
        return MODE if _pd.nslice_center_capable(g, b) else 0
    return _pd.nslice_center_mode(g, b)


for _name in ("conv2d_3x3", "prepare_conv2d_bias", "prepare_conv2d_weights"):
    setattr(_ut, _name, getattr(_op, _name))
_ut._op_module = _op
_ut._pd = _pd


@pytest.fixture(autouse=True)
def _candidate(monkeypatch):
    if _SUB == "op_copy":
        monkeypatch.setattr(_pd, "NSLICE_CENTER", MODE)
    for k, v in KNOBS.items():
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
test_conv2d_3x3_block_sharded_grid_column_mismatch = _ut.test_conv2d_3x3_block_sharded_grid_column_mismatch
test_conv2d_3x3_block_sharded_writer_only_scatter = _ut.test_conv2d_3x3_block_sharded_writer_only_scatter


# ((H, W, C_in), C_out, padding = dilation), shard (rows, width), extra kwargs
EXTRA = [
    # d = 4 halo several shards away, 2-tile slices, ragged last row core (M = 1536 px over 7 rows of 224)
    (((24, 64, 128), 96, 4), (224, 64)),
    # aspp-like short last slice: C_in 160 = 2 x 64 + 32 (TILE short slice of whole tiles), d = 6
    (((16, 64, 160), 64, 6), (128, 64)),
    # d = 1, 1-tile slices, W = 32 (one tile-row per image row)
    (((16, 32, 128), 64, 1), (64, 32)),
    # d = 2, 3-tile slices, C_out 160: short last N-slice
    (((16, 64, 288), 160, 2), (256, 96)),
]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias_relu", "plain"])
@pytest.mark.parametrize("dtype", ["bf8b", "bf16"])
@pytest.mark.parametrize("case", EXTRA, ids=[f"H{c[0][0][0]}_W{c[0][0][1]}_Ci{c[0][0][2]}_d{c[0][2]}" for c in EXTRA])
def test_nct_extra(case, dtype, with_bias, device, monkeypatch):
    import ttnn

    real_select = _pd.select_blocking
    seen = []

    def recording(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        seen.append(_mode(g, b))
        return b

    monkeypatch.setattr(_op, "select_blocking", recording)

    ((H, W, c_in), c_out, pdil), shard = case
    dt = ttnn.bfloat8_b if dtype == "bf8b" else ttnn.bfloat16
    mc = _ut._block_config(H * W, c_in, ttnn.TILE_LAYOUT, shard)
    grid = device.compute_with_storage_grid_size()
    rows, cols = _pd.block_grid_shape(mc.shard_spec)
    if cols > grid.x or rows > grid.y:
        pytest.skip("grid")
    _ut._run(
        ((H, W, c_in), c_out, pdil, 1, pdil),
        dt,
        ttnn.TILE_LAYOUT,
        mc,
        device,
        with_bias=with_bias,
        relu=with_bias,
        dispatches=2,
    )
    assert seen and all(m == MODE for m in seen), seen


@pytest.mark.parametrize("bm", [1, 2], ids=["bm1", "bm2"])
def test_nct_multi_m_block(bm, device, monkeypatch):
    """Several M-blocks per core on the N-slice walk (d = 2, 2-tile slices): cb_center_tiles is reserved, copied, pushed
    and popped per M-block, and the copy's rows follow the block."""
    import torch
    import ttnn

    torch.manual_seed(3)
    (H, W, c_in), c_out = (16, 64, 128), 64
    mc = _ut._block_config(H * W, c_in, ttnn.TILE_LAYOUT, (128, 64))
    real_select = _pd.select_blocking
    seen = []

    def small_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.k_split and b.core_m_tiles == 4 and b.nslice
        fit = _pd._footprint(g, b.core_m_tiles, bm, b.block_k_tiles, b.block_n_tiles, nslice=True)
        nb = _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=bm,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )
        seen.append(_mode(g, nb))
        return nb

    monkeypatch.setattr(_op, "select_blocking", small_blocks)
    _ut._run(
        ((H, W, c_in), c_out, 2, 1, 2),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        mc,
        device,
        with_bias=True,
        relu=True,
        dispatches=2,
    )
    assert seen and all(m == MODE for m in seen), seen
