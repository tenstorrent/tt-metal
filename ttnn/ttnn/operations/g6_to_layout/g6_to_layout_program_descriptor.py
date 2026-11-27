# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout program descriptor: TILE bfloat16 -> ROW_MAJOR bfloat16, interleaved, dataflow only.

Two regimes, each its own program of a reader (NCRISC, NoC0) and a writer (BRISC, NoC1); no
value ever enters the Tensix coprocessor, so every bf16 bit pattern survives:

    staged_reorder (R1): live face-row prefixes -> cb_tile_window -> 32-B self-aimed copies
                         -> cb_row_block page -> one write per live row segment.
    face_direct    (R2): one tile column with W <= 16: faces 0/2 prefixes -> cb_left_faces
                         -> one write of 2*W bytes per row straight from its face row.

Work unit: one tile of the flattened (tile_row, tile_col) space, split over the grid in
contiguous row-major tile ranges. Blocks are rectangles cut from a core's range by the walk in
kernels/g6_to_layout_block_walk.hpp. Every CB size and kernel arg derives from the knobs below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Block and buffer-depth knobs (single source of truth).
READ_BATCH_TILES = 8  # tiles per DRAM read batch in R1
TILE_WINDOW_DEPTH = 2  # R1 batches resident in cb_tile_window
ROW_DEPTH = 2  # blocks resident in cb_row_block / cb_left_faces
# Target blocks per core; sets block_rows_tiles below its L1 cap so pipeline fill/drain stays near
# 1/17 of a core's run. Measured against {1 (the L1-cap block), 4, 64}: 5-6% faster than the L1-cap
# block on cases 304 and 1978, 6% faster than 64 on 1978, and within run-to-run spread on 308 (whose
# time is set by DRAM write transactions, not by the block). The zone build spends 4 profiler markers
# per block, so 16 + 2 ragged blocks also stay well inside the 250-marker buffer of a RISC.
PIPELINE_BLOCKS_PER_CORE = 16

# L1 per core the model leaves to the op (BB-05); an L1 output's pages on a core count inside it.
L1_BUDGET_BYTES = 1347776

TILE_HW = 32
FACE_HW = 16
DATUM_BYTES = 2  # bfloat16
TILE_BYTES = TILE_HW * TILE_HW * DATUM_BYTES
LEFT_SLOT_BYTES = TILE_BYTES // 2  # faces 0 and 2 of one tile
TILE_ROW_BYTES = TILE_HW * DATUM_BYTES  # bytes one tile column adds to an output row

STAGED_REORDER = "staged_reorder"
FACE_DIRECT = "face_direct"

CB_TILE_WINDOW = 0
CB_ROW_BLOCK = 1
CB_LEFT_FACES = 2

# The R1 reader tags DRAM batches with trids 1..TILE_WINDOW_DEPTH and alternates two more over
# the reorder copies; ids are 4 bits and 0 means untagged.
_REORDER_TRIDS = 2
_MAX_TRID = 15
_MAX_TRID_OUTSTANDING = 255
assert TILE_WINDOW_DEPTH + _REORDER_TRIDS <= _MAX_TRID
assert 2 * READ_BATCH_TILES <= _MAX_TRID_OUTSTANDING  # at most 2 reads per tile per batch


def _div_up(a, b):
    return -(-a // b)


def _round_up(a, b):
    return _div_up(a, b) * b


@dataclass(frozen=True)
class Geometry:
    images: int
    height: int
    width: int
    image_rows_tiles: int
    tensor_cols_tiles: int

    @property
    def tensor_rows_tiles(self):
        return self.images * self.image_rows_tiles

    @property
    def tensor_tiles(self):
        return self.tensor_rows_tiles * self.tensor_cols_tiles

    @property
    def out_pages(self):
        return self.images * self.height


def tensor_geometry(shape):
    """Leading dims fold into N; each image is tile-padded on its own."""
    images = 1
    for d in shape[:-2]:
        images *= d
    height, width = shape[-2], shape[-1]
    return Geometry(images, height, width, _div_up(height, TILE_HW), _div_up(width, TILE_HW))


def select_regime(geometry):
    if geometry.tensor_cols_tiles == 1 and geometry.width <= FACE_HW:
        return FACE_DIRECT
    return STAGED_REORDER


def _tile_row_stage_rows(geometry):
    return min(TILE_HW, geometry.height)


def _window_bytes():
    return TILE_WINDOW_DEPTH * READ_BATCH_TILES * TILE_BYTES


def _block_bytes(regime, block_rows_tiles, block_cols_tiles, stage_rows):
    if regime == FACE_DIRECT:
        return block_rows_tiles * LEFT_SLOT_BYTES
    return block_rows_tiles * stage_rows * block_cols_tiles * TILE_ROW_BYTES


def _max_block_tiles(regime, stage_rows, out_l1_share):
    if regime == FACE_DIRECT:
        return (L1_BUDGET_BYTES - out_l1_share) // (ROW_DEPTH * LEFT_SLOT_BYTES)
    return (L1_BUDGET_BYTES - out_l1_share - _window_bytes()) // (ROW_DEPTH * stage_rows * TILE_ROW_BYTES)


def _choose_blocking(geometry, core_tiles_max, max_block_tiles):
    """(block_rows_tiles, block_cols_tiles): the whole row when it fits, else the core's column
    span; full-row blocks get tall enough for ~PIPELINE_BLOCKS_PER_CORE blocks per core."""
    cols = geometry.tensor_cols_tiles
    block_cols_tiles = min(cols, core_tiles_max, max_block_tiles)
    if block_cols_tiles < cols:
        return 1, block_cols_tiles
    core_full_rows_max = core_tiles_max // cols
    rows = _div_up(core_full_rows_max, PIPELINE_BLOCKS_PER_CORE)
    return max(1, min(rows, max_block_tiles // block_cols_tiles)), block_cols_tiles


@dataclass
class Plan:
    regime: str
    geometry: Geometry
    grid: tuple
    num_cores: int
    all_cores: object
    core_ranges: list  # [((x, y), tile_begin, num_tiles)] in split order
    block_rows_tiles: int
    block_cols_tiles: int
    tile_row_stage_rows: int
    cb_bytes: dict  # CB index -> total bytes
    out_l1_share: int

    @property
    def l1_bytes_per_core(self):
        return sum(self.cb_bytes.values()) + self.out_l1_share

    def report(self):
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "grid": self.grid,
            "block_rows_tiles": self.block_rows_tiles,
            "block_cols_tiles": self.block_cols_tiles,
            "tile_row_stage_rows": self.tile_row_stage_rows,
            "read_batch_tiles": READ_BATCH_TILES if self.regime == STAGED_REORDER else None,
            "tile_window_depth": TILE_WINDOW_DEPTH if self.regime == STAGED_REORDER else None,
            "row_depth": ROW_DEPTH,
            "cb_bytes": dict(self.cb_bytes),
            "out_l1_share": self.out_l1_share,
            "l1_bytes_per_core": self.l1_bytes_per_core,
            "cores": [
                {"core": core, "roles": ("reader", "writer"), "tile_begin": begin, "num_tiles": count}
                for core, begin, count in self.core_ranges
            ],
        }


def compute_plan(shape, grid, *, output_in_l1, num_l1_banks, out_alignment):
    """Regime, core split and block extents of one call. Shape-only regime selection."""
    geometry = tensor_geometry(shape)
    regime = select_regime(geometry)
    stage_rows = _tile_row_stage_rows(geometry)
    grid_x, grid_y = grid

    out_l1_share = 0
    if output_in_l1:
        out_aligned_page_bytes = _round_up(geometry.width * DATUM_BYTES, out_alignment)
        banks_per_core = max(1, _div_up(num_l1_banks, grid_x * grid_y))
        out_l1_share = _div_up(geometry.out_pages, num_l1_banks) * out_aligned_page_bytes * banks_per_core

    max_block_tiles = _max_block_tiles(regime, stage_rows, out_l1_share)
    if max_block_tiles < 1:
        raise RuntimeError(
            f"g6_to_layout: an L1 output of {out_l1_share} B per core leaves no room for one block "
            f"within the {L1_BUDGET_BYTES} B L1 budget"
        )

    num_cores, all_cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(
        ttnn.CoreCoord(grid_x, grid_y), geometry.tensor_tiles, row_wise=True
    )
    core_ranges = []
    tile = 0
    for group, count in ((group_1, tiles_1), (group_2, tiles_2)):
        if count == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            core_ranges.append(((core.x, core.y), tile, count))
            tile += count
    assert tile == geometry.tensor_tiles, f"g6_to_layout: split covered {tile} of {geometry.tensor_tiles} tiles"

    block_rows_tiles, block_cols_tiles = _choose_blocking(geometry, max(tiles_1, tiles_2), max_block_tiles)
    block_bytes = _block_bytes(regime, block_rows_tiles, block_cols_tiles, stage_rows)
    if regime == FACE_DIRECT:
        cb_bytes = {CB_LEFT_FACES: ROW_DEPTH * block_bytes}
    else:
        cb_bytes = {CB_TILE_WINDOW: _window_bytes(), CB_ROW_BLOCK: ROW_DEPTH * block_bytes}

    return Plan(
        regime=regime,
        geometry=geometry,
        grid=(grid_x, grid_y),
        num_cores=num_cores,
        all_cores=all_cores,
        core_ranges=core_ranges,
        block_rows_tiles=block_rows_tiles,
        block_cols_tiles=block_cols_tiles,
        tile_row_stage_rows=stage_rows,
        cb_bytes=cb_bytes,
        out_l1_share=out_l1_share,
    )


def _cb(index, total_bytes, page_bytes, core_ranges):
    return ttnn.CBDescriptor(
        total_size=total_bytes,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page_bytes)
        ],
    )


class G6ToLayoutProgram:
    """Program descriptor of one plan, built once. A later call with the same cache key only
    rebinds the two buffer addresses (common runtime arg 0 of the reader and of the writer)."""

    def __init__(self, plan, input_tensor, output_tensor):
        g = plan.geometry
        all_cores = plan.all_cores
        geometry_ct_args = [
            g.tensor_cols_tiles,
            g.image_rows_tiles,
            g.height,
            g.width,
            plan.block_rows_tiles,
            plan.block_cols_tiles,
            plan.tile_row_stage_rows,
        ]
        block_bytes = _block_bytes(plan.regime, plan.block_rows_tiles, plan.block_cols_tiles, plan.tile_row_stage_rows)

        if plan.regime == FACE_DIRECT:
            cbs = [_cb(CB_LEFT_FACES, plan.cb_bytes[CB_LEFT_FACES], block_bytes, all_cores)]
            reader_source, writer_source = "g6_to_layout_face_reader.cpp", "g6_to_layout_face_writer.cpp"
            reader_ct_args = geometry_ct_args + [CB_LEFT_FACES]
            writer_ct_args = geometry_ct_args + [CB_LEFT_FACES]
        else:
            cbs = [
                _cb(CB_TILE_WINDOW, plan.cb_bytes[CB_TILE_WINDOW], TILE_BYTES, all_cores),
                _cb(CB_ROW_BLOCK, plan.cb_bytes[CB_ROW_BLOCK], block_bytes, all_cores),
            ]
            reader_source, writer_source = "g6_to_layout_staged_reader.cpp", "g6_to_layout_staged_writer.cpp"
            reader_ct_args = geometry_ct_args + [CB_TILE_WINDOW, CB_ROW_BLOCK, READ_BATCH_TILES, TILE_WINDOW_DEPTH]
            writer_ct_args = geometry_ct_args + [CB_ROW_BLOCK]
        reader_ct_args += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        writer_ct_args += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

        core_rt_args = ttnn.RuntimeArgs()
        for (x, y), tile_begin, num_tiles in plan.core_ranges:
            core_rt_args[x][y] = [tile_begin, num_tiles]

        self._reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / reader_source),
            core_ranges=all_cores,
            compile_time_args=reader_ct_args,
            runtime_args=core_rt_args,
            common_runtime_args=[input_tensor.buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
        )
        self._writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / writer_source),
            core_ranges=all_cores,
            compile_time_args=writer_ct_args,
            runtime_args=core_rt_args,
            common_runtime_args=[output_tensor.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
        )
        self._descriptor = ttnn.ProgramDescriptor(kernels=[self._reader, self._writer], semaphores=[], cbs=cbs)

    def descriptor_for(self, input_tensor, output_tensor):
        self._reader.common_runtime_args = [input_tensor.buffer_address()]
        self._writer.common_runtime_args = [output_tensor.buffer_address()]
        self._descriptor.kernels = [self._reader, self._writer]
        return self._descriptor
