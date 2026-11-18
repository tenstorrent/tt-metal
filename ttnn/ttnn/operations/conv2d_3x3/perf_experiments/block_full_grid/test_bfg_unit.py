# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness: run a conv2d_3x3 unit-test file's tests against this experiment's op copy with a variant's knobs
(BFG_UNIT_VARIANT, default `extend`; the knob sets of test_bfg.VARIANTS).

The unit file's source is loaded with its op imports pointed at the copy and executed into this module, so pytest
collects its tests unchanged. BFG_UNIT picks the file (block | height | base | layouts).

  BFG_UNIT=block scripts/run_safe_pytest.sh --run-all \\
      ttnn/ttnn/operations/conv2d_3x3/perf_experiments/block_full_grid/test_bfg_unit.py --import-mode=prepend
"""

import importlib
import os
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[5]
_COPY = os.environ.get("BFG_PKG", "op_copy")  # grad: the graduation copy (no knobs)
PKG = "ttnn.operations.conv2d_3x3.perf_experiments.block_full_grid." + _COPY
_pd = importlib.import_module(PKG + ".conv2d_3x3_program_descriptor")

_VARIANTS_SRC = (HERE / "test_bfg.py").read_text()
_ns = {}
# Only the knob tables (no ttnn / device code): KNOB_DEFAULTS and VARIANTS.
exec(
    compile(
        _VARIANTS_SRC[_VARIANTS_SRC.index("KNOB_DEFAULTS = dict(") : _VARIANTS_SRC.index("def apply_variant")],
        "test_bfg.py",
        "exec",
    ),
    _ns,
)
_variant = os.environ.get("BFG_UNIT_VARIANT", "extend")
if _COPY != "grad":
    for _k, _v in {**_ns["KNOB_DEFAULTS"], **_ns["VARIANTS"][_variant]}.items():
        setattr(_pd, _k, _v)
_pd.select_blocking.cache_clear()

_UNIT_FILES = {
    "height": "test_conv2d_3x3_height_sharded.py",
    "block": "test_conv2d_3x3_block_sharded.py",
    "base": "test_conv2d_3x3.py",
    "layouts": "test_conv2d_3x3_layouts.py",
}
_src_path = REPO / "tests/ttnn/unit_tests/operations/conv2d_3x3" / _UNIT_FILES[os.environ.get("BFG_UNIT", "block")]
if os.environ.get("BFG_UNIT_SRC"):
    # An updated copy of the unit file (grad_tests/: the tests graduation.patch needs).
    _src_path = HERE / os.environ["BFG_UNIT_SRC"]
_src = _src_path.read_text()
_n = 0
for old, new in [
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"', f'"{PKG}.conv2d_3x3_program_descriptor"'),
    ('"ttnn.operations.conv2d_3x3.conv2d_3x3"', f'"{PKG}.conv2d_3x3"'),
    ("from ttnn.operations.conv2d_3x3 import", f"from {PKG}.conv2d_3x3 import"),
]:
    _n += _src.count(old)
    _src = _src.replace(old, new)
assert _n > 0
assert "ttnn.operations.conv2d_3x3 import" not in _src.replace(f"{PKG}.conv2d_3x3 import", "")
assert '"ttnn.operations.conv2d_3x3.conv2d_3x3' not in _src, "an op import still points at the real op"
exec(compile(_src, str(_src_path), "exec"), globals())

if os.environ.get("BFG_UNIT", "block") == "block" and not os.environ.get("BFG_UNIT_SRC"):
    # The unit file's multi-M-block and ragged-row tests pin the shard-row split (core_m_tiles 4 / 7), which the
    # extended rows change. These two run the same mechanisms where the rows ARE extended (remote output).

    @pytest.mark.parametrize("dtype,layout,with_bias", MULTI_BLOCK_CASES)
    def test_bfg_multi_m_block_extended_rows(dtype, layout, with_bias, device, monkeypatch):
        """One-tile-row M-blocks on extended rows: 20 output tile-rows on 3 shard rows of 7 run on 10 grid rows of 2,
        so every core runs two combine rounds and stores two blocks to output shards it does not sit on."""
        torch.manual_seed(42)
        (H, W, c_in), c_out = (16, 40, 64), 96
        shard = (224, 32) if layout == ttnn.TILE_LAYOUT else (224, 16)
        config = _block_config(H * W, c_in, layout, shard)
        real_select = _pd.select_blocking

        def one_tile_row_blocks(g, grid, l1_budget):
            b = real_select(g, grid, l1_budget)
            assert g.k_split and g.remote_output and grid[0] == 10 and b.core_m_tiles == 2, (grid, b)
            fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
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

    @pytest.mark.parametrize("dtype,layout", LAYOUTS)
    def test_bfg_column_weight_mcast_ragged_extended_row(dtype, layout, device, monkeypatch):
        """The ragged-row test on extended rows: the 3-row shard grid starts at y = 3, so 7 grid rows reach the device
        edge; 20 output tile-rows split 3 per row, the last row runs 2 one-tile-row M-blocks (6 x 3 + 2) and streams its
        own weights while the 6 full rows share each column's multicast."""
        torch.manual_seed(5)
        (H, W, c_in), c_out = (16, 40, 64), 64
        shard = (224, 32) if layout == ttnn.TILE_LAYOUT else (224, 16)
        config = _block_config(H * W, c_in, layout, shard, start=(0, 3))
        real_select = _pd.select_blocking
        real_column_mcast = _pd._column_weight_mcast
        channels = []

        def one_tile_row_blocks(g, grid, l1_budget):
            b = real_select(g, grid, l1_budget)
            assert g.k_split and g.remote_output and grid[0] == 7 and b.core_m_tiles == 3 and b.num_m_groups == 7
            fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
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

        def recording_column_mcast(input_tensor, work, m_blocks_per_index, **kwargs):
            channel = real_column_mcast(input_tensor, work, m_blocks_per_index, **kwargs)
            channels.append((channel, list(work)))
            return channel

        monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
        monkeypatch.setattr(_pd, "_column_weight_mcast", recording_column_mcast)
        _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=True, relu=True)

        ((channel, work),) = channels
        assert channel is not None, "the column weight multicast channel was not built"
        receivers, group_size, _ = channel
        k_cols = _pd.block_grid_shape(config.shard_spec)[1]
        assert group_size == 6 and receivers.num_cores() == 6 * k_cols
        assert not any(receivers.contains(core) for core, row, _ in work if row == 6), "the ragged row streams its own"
