# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3 on an L1 HEIGHT_SHARDED input (regime R2): output HEIGHT_SHARDED on the input's shard grid.

- layout matrix: ROW_MAJOR bf16, TILE bf16 and TILE bf8_b with shards small enough that one core's
  halo comes from several other cores, and ROW_MAJOR shards that do not start on an image row.
- placement: the output shard spec, a COL_MAJOR shard orientation on a grid that does not start at
  (0, 0) (work index i must run where shard i lives), and the program-cache hit on fresh shards.
- multi-M-block shard: a pinned M-block smaller than the shard makes each core pack several M-blocks
  into its output shard, with ReLU state crossing the blocks.
- weight multicast (regime R1): a ragged last core that runs fewer M-blocks stays outside the
  channel and streams its own weights, while the others take one multicast round per K-block.
- face-direct im2col (stride * C_in == 16, ROW_MAJOR): the reader and the writer gather alternate in0 row
  groups straight into tile faces, the writer reads the slab runs of higher-index shards, and compute
  multiplies (and adds the bias to) each group as it lands; also with several M-blocks per shard.
"""

import importlib

import pytest


class _LazyTorch:
    def __getattr__(self, name):
        import torch as _torch

        return getattr(_torch, name)


torch = _LazyTorch()
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple.graduated.conv2d_3x3 import (
    conv2d_3x3,
    prepare_conv2d_bias,
    prepare_conv2d_weights,
)

_op_module = importlib.import_module(
    "ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple.graduated.conv2d_3x3"
)
_pd = importlib.import_module(
    "ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple.graduated.conv2d_3x3_program_descriptor"
)

PCC = {ttnn.bfloat16: 0.997, ttnn.bfloat8_b: 0.99}

LAYOUTS = [
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="rm_bf16"),
    pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="tile_bf16"),
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="tile_bf8b"),
]

# ((H, W, C_in), C_out, padding, stride, dilation, shard rows per core)
SHAPES = [
    ((16, 16, 64), 128, 1, 1, 1, 64),  # 4 shards; each core's halo is a neighbour's full shard
    ((32, 32, 8), 32, 1, 2, 1, 96),  # stem-like: C_in = 8, stride 2
    ((32, 64, 64), 64, 8, 1, 8, 64),  # d = 8: the halo reaches 8 shards away
    ((16, 34, 48), 64, 0, 1, 1, 160),  # valid padding, C_in = 48 (lanes in the second channel tile)
]
RM_SHARD_ROWS = {64: 40, 96: 70, 160: 136}  # ROW_MAJOR shards: not a multiple of an image row


def _case_id(case):
    (H, W, c_in), c_out, p, s, d, rows = case
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{p}_s{s}_d{d}_shard{rows}"


def _row_major_grid(num_cores, grid_x):
    """`num_cores` cores filled row-major from (0, 0): full rows, then a partial last row."""
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _height_config(c_in, layout, shard_rows, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    width = -(-c_in // 32) * 32 if layout == ttnn.TILE_LAYOUT else c_in
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_rows, width], orientation),
    )


def _row_major_config(device, num_px, c_in, layout, shard_rows):
    grid_x = device.compute_with_storage_grid_size().x
    return _height_config(c_in, layout, shard_rows, _row_major_grid(-(-num_px // shard_rows), grid_x))


def _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, relu):
    c_in = torch_input.shape[-1]
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    out = torch.nn.functional.conv2d(
        x, torch_weight.float(), None if torch_bias is None else torch_bias.float(), stride=s, padding=p, dilation=d
    )
    if relu:
        out = torch.relu(out)
    return out.permute(0, 2, 3, 1).reshape(1, 1, -1, out.shape[1])


def _run(case_hw, dtype, layout, memory_config, device, *, with_bias, relu, dispatches=1):
    (H, W, c_in), c_out, p, s, d = case_hw
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if with_bias else None
    kwargs = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=memory_config,
        device=device,
    )
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs) if with_bias else None
    h_out = (H + 2 * p - 2 * d - 1) // s + 1
    w_out = (W + 2 * p - 2 * d - 1) // s + 1
    m = h_out * w_out
    num_cores = memory_config.shard_spec.num_cores()

    live = []  # keep every dispatch's tensors alive, so later dispatches get fresh addresses
    for _ in range(dispatches):
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        tt_output = conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=p,
            stride=s,
            dilation=d,
            activation="relu" if relu else None,
        )
        live.append((tt_input, tt_output))

        out_config = tt_output.memory_config()
        assert out_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        assert out_config.buffer_type == ttnn.BufferType.L1
        out_spec = out_config.shard_spec
        assert out_spec.grid == memory_config.shard_spec.grid
        assert out_spec.orientation == memory_config.shard_spec.orientation
        out_rows = -(-(-(-m // num_cores)) // 32) * 32  # roundup32(ceil(M / n_cores))
        assert list(out_spec.shape) == [out_rows, -(-c_out // 32) * 32]
        assert tt_output.layout == ttnn.TILE_LAYOUT and tt_output.dtype == dtype
        assert list(tt_output.shape) == [1, 1, m, c_out]

        actual = ttnn.to_torch(tt_output).float()
        expected = _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, relu)
        assert torch.isfinite(actual).all()
        assert_with_pcc(expected, actual, PCC[ttnn.bfloat8_b])
        if relu:
            assert (actual >= 0).all()


@pytest.mark.parametrize("with_bias,relu", [(True, True), (False, False)], ids=["bias_relu", "plain"])
@pytest.mark.parametrize("dtype,layout", LAYOUTS)
@pytest.mark.parametrize("case", SHAPES, ids=[_case_id(c) for c in SHAPES])
def test_conv2d_3x3_height_sharded(case, dtype, layout, with_bias, relu, device):
    torch.manual_seed(42)
    *case_hw, shard_rows = case
    H, W, c_in = case_hw[0]
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        shard_rows = RM_SHARD_ROWS[shard_rows]
    config = _row_major_config(device, H * W, c_in, layout, shard_rows)
    _run(tuple(case_hw), dtype, layout, config, device, with_bias=with_bias, relu=relu)


@pytest.mark.parametrize("dtype,layout", LAYOUTS)
def test_conv2d_3x3_height_sharded_col_major_offset_grid(dtype, layout, device):
    """COL_MAJOR shard order on a 3x4 grid starting at (1, 1), 11 shards on 12 cores, dispatched again on
    fresh shards: work index i must run on shard core i and the re-dispatch must hit the program cache."""
    torch.manual_seed(7)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 4))])
    shard_rows = 96 if layout == ttnn.TILE_LAYOUT else 100
    config = _height_config(32, layout, shard_rows, grid, ttnn.ShardOrientation.COL_MAJOR)
    case_hw = ((32, 32, 32), 64, 1, 1, 1)
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=False)
    n_programs = device.num_program_cache_entries()
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=False, dispatches=2)
    assert device.num_program_cache_entries() == n_programs, "re-dispatch on fresh shards missed the program cache"


MULTI_BLOCK_CASES = [
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True, id="tile_bf8b_bias_relu"),
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, False, id="rm_bf16_relu"),
]


@pytest.mark.parametrize("dtype,layout,with_bias", MULTI_BLOCK_CASES)
def test_conv2d_3x3_height_sharded_multi_m_block(dtype, layout, with_bias, device, monkeypatch):
    """A one-tile-row M-block inside a 4-tile-row output shard: compute packs four M-blocks into the
    shard in order, and PACK_RELU / packer L1-acc state must not leak from one block into the next."""
    torch.manual_seed(42)
    (H, W, c_in), c_out = (32, 64, 32), 32
    shard_rows = 128 if layout == ttnn.TILE_LAYOUT else 125
    config = _row_major_config(device, H * W, c_in, layout, shard_rows)
    real_select = _pd.select_blocking

    def one_tile_row_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.sharded_output and b.core_m_tiles == 4, "the test expects a 4-tile-row output shard"
        fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
    _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=with_bias, relu=True)


@pytest.mark.parametrize("dtype,layout", [LAYOUTS[0], LAYOUTS[2]])
def test_conv2d_3x3_height_sharded_weight_mcast_ragged_core(dtype, layout, device, monkeypatch):
    """One-tile-row M-blocks in 6-tile-row output shards over 64 tile-rows: 10 cores run 6 M-blocks
    (18 multicast rounds of 3 kh K-blocks), the last runs 4 and must read its own weights. A ragged core
    inside the channel would take rounds it never consumes and desynchronize every cb_weights slot."""
    torch.manual_seed(3)
    (H, W, c_in), c_out = (32, 64, 64), 64
    config = _row_major_config(device, H * W, c_in, layout, 192)
    real_select = _pd.select_blocking
    real_mcast_cores = _pd._weight_mcast_cores
    channels = []

    def one_tile_row_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.sharded_output and b.core_m_tiles == 6 and b.num_m_groups == 11
        fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles)
        assert fit["num_k_blocks"] == 3
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    def recording_mcast_cores(core_list, m_blocks_per_core):
        receivers = real_mcast_cores(core_list, m_blocks_per_core)
        channels.append((receivers, list(core_list)))
        return receivers

    monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
    monkeypatch.setattr(_pd, "_weight_mcast_cores", recording_mcast_cores)
    _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=True, relu=True)

    ((receivers, core_list),) = channels
    assert receivers is not None, "the weight multicast channel was not built"
    assert receivers.num_cores() == len(core_list) - 1
    assert not receivers.contains(core_list[-1]), "the ragged last core must stream its own weights"


# ((H, W, C_in), C_out, padding, stride, dilation, ROW_MAJOR shard rows): every core's output is several tile-rows,
# so in0 streams in several row groups split between the reader and the writer, and shards that do not start on an
# image row put part of each footprint in higher-index shards.
FACE_DIRECT_SHAPES = [
    ((64, 128, 8), 64, 1, 2, 1, 1000),  # stem-like: two taps per face row; 8 tile-rows per core
    ((32, 64, 16), 32, 1, 1, 1, 230),  # one tap per face row; the fourth face column is K padding only
    ((48, 64, 8), 64, 1, 2, 1, 700),  # 3 tile-rows per core: an odd row-group count, a shorter last group
]


def _record_blocking(monkeypatch):
    real_select = _pd.select_blocking
    seen = []

    def recording_select(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        seen.append((g, b))
        return b

    monkeypatch.setattr(_op_module, "select_blocking", recording_select)
    return seen


@pytest.mark.parametrize(
    "with_bias,relu", [(True, True), (False, True), (True, False)], ids=["bias_relu", "relu", "bias"]
)
@pytest.mark.parametrize("case", FACE_DIRECT_SHAPES, ids=[_case_id(c) for c in FACE_DIRECT_SHAPES])
def test_conv2d_3x3_height_sharded_face_direct(case, with_bias, relu, device, monkeypatch):
    torch.manual_seed(11)
    *case_hw, shard_rows = case
    H, W, c_in = case_hw[0]
    config = _row_major_config(device, H * W, c_in, ttnn.ROW_MAJOR_LAYOUT, shard_rows)
    seen = _record_blocking(monkeypatch)
    _run(tuple(case_hw), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, config, device, with_bias=with_bias, relu=relu)
    ((g, b),) = seen
    assert _pd.face_direct_im2col(g, b.block_k_tiles) and b.num_k_blocks == 1
    assert b.in0_stream_rows < b.block_m_tiles, "the test expects several in0 row groups per M-block"


def test_conv2d_3x3_height_sharded_face_direct_multi_m_block(device, monkeypatch):
    """Two M-blocks per shard on the face-direct path: the writer refills its slab runs only after the reader
    released the first block's slab, and the gather/push counters carry across blocks."""
    torch.manual_seed(5)
    *case_hw, shard_rows = FACE_DIRECT_SHAPES[0]
    H, W, c_in = case_hw[0]
    config = _row_major_config(device, H * W, c_in, ttnn.ROW_MAJOR_LAYOUT, shard_rows)
    real_select = _pd.select_blocking

    def two_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.sharded_output and b.core_m_tiles == 8
        fit = _pd._footprint(g, b.core_m_tiles, 4, b.block_k_tiles, b.block_n_tiles)
        assert fit["num_k_blocks"] == 1 and fit["in0_stream_rows"] < 4
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=4,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(_op_module, "select_blocking", two_blocks)
    _run(tuple(case_hw), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, config, device, with_bias=True, relu=True)
