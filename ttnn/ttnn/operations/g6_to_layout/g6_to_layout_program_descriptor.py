# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_to_layout program descriptor: TILE <-> ROW_MAJOR.

Six regimes, each its own program. The bf16 regimes are a reader (NCRISC, NoC0) and a writer
(BRISC, NoC1) only; no value enters the Tensix coprocessor, so every bf16 bit pattern survives:

    staged_reorder   (R1): live face-row prefixes -> cb_tile_window -> 32-B self-aimed copies
                           -> cb_row_block page -> one write per live row segment.
    face_direct      (R2): one tile column with W <= 16: faces 0/2 prefixes -> cb_left_faces
                           -> one write of 2*W bytes per row straight from its face row.
    same_bank_slot_writes (R7): R2 whose 2*W-byte row is shorter than its aligned page slot: the
                           writer copies each row into its slot of a per-bank image (pages p,
                           p+B, p+2B... of a bank are consecutive slots) and writes each bank's
                           run of slots in one write, alignment gaps included.

A bfloat8_b input is a value conversion, so it adds a compute kernel between the same two CBs:

    packer_untilize  (R3): whole bfp8 tiles -> cb_tile_window -> unpack + pack-untilize to bf16
                           -> cb_row_block tile pages -> one write per live row segment. The
                           packer renders NaN datums as +-inf, so the reader marks them on
                           cb_nan_marks and the writer stores the host's NaN (0xFFFF) over them.
    sharded_in_place (R4): R3 with cb_tile_window backed on the resident shard (zero-copy); the
                           core assignment and the block are the caller's shard.

A ROW_MAJOR bfloat16 input mirrors R1/R3 with the transfer shapes swapped:

    row_major_to_tile (R5): one read per live row segment -> cb_row_stage tile-row slots ->
                           32-B self-aimed copies into tile slots -> cb_tile_block -> whole-tile
                           writes. Dataflow-only, so bit-exact for every bf16 pattern.
    packer_tilize    (R6): bfloat8_b output: the row segments land in cb_row_stage, compute
                           tilizes and packs bfp8 into cb_tile_block, the R5 writer writes tiles.

Interleaved work unit: one tile of the flattened (tile_row, tile_col) space, split over the grid
in contiguous row-major tile ranges. Blocks are rectangles cut from a core's range (or its shard)
by the walks in kernels/g6_to_layout_block_walk.hpp. Every CB size and kernel arg derives from the
knobs below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Block and buffer-depth knobs (single source of truth).
READ_BATCH_TILES = 8  # tiles per DRAM read batch in R1
TILE_WINDOW_DEPTH = 2  # R1 batches resident in cb_tile_window
ROW_DEPTH = 2  # blocks resident in cb_row_block / cb_left_faces / cb_tile_block
# R7: blocks whose face reads are on the wire (one trid each) while the reader builds its share of
# the oldest landed block, and the most it issues when the next block has still not landed after the
# build (equal: no adaptive read-ahead).
SLOT_READ_DEPTH = 1
SLOT_READ_DEPTH_MAX = 1
# R7 bank_stagger: each core starts every block's reads at the input bank of its own (x + y), not at the
# block's first tile.
SLOT_READ_STAGGER = False
# R7 cb_left_faces / cb_slot_image depths. Besides the blocks in flight, cb_left_faces holds the block
# the reader builds and the block k - 1 the writer still builds; the image ring holds k - 2 (still
# draining), k - 1 and k. Below that the two halves of consecutive blocks serialize.
SLOT_FACES_DEPTH = SLOT_READ_DEPTH_MAX + 2
SLOT_IMAGE_DEPTH = 3
# R7: eighths of each block's rows the reader moves into the slot image with RISC loads/stores (while
# the next block's reads are in flight); the writer moves the rest, with self-aimed NoC reads when
# WRITER_SLOT_NOC_COPIES (the two RISCs' loads/stores share one path into L1, so RISC copies on both
# do not add up), else RISC loads/stores. 0: the writer builds the whole image.
READER_SLOT_EIGHTHS = 5
WRITER_SLOT_NOC_COPIES = True
ROW_WINDOW_DEPTH = 2  # R5 tile-row loads resident in cb_row_stage
# R5: the writer copies the bottom faces of each staged tile-row (its NoC, its RISC) while the reader
# copies the top faces; off, the reader copies both and cb_row_stage stays reader-private.
SPLIT_REORDER = True
TILE_BLOCK_DEPTH = 2  # R3 blocks resident in cb_tile_window (reader -> compute)
# cb_nan_marks pages: the reader can run TILE_BLOCK_DEPTH + ROW_DEPTH blocks ahead of the writer.
NAN_MARKS_DEPTH = TILE_BLOCK_DEPTH + ROW_DEPTH
# Target blocks per core; sets block_rows_tiles below its L1 cap so pipeline fill/drain stays near
# 1/17 of a core's run. Measured against {1 (the L1-cap block), 4, 64}: 5-6% faster than the L1-cap
# block on cases 304 and 1978, 6% faster than 64 on 1978, and within run-to-run spread on 308 (whose
# time is set by DRAM write transactions, not by the block). The zone build spends 4 profiler markers
# per block, so 16 + 2 ragged blocks also stay well inside the 250-marker buffer of a RISC.
PIPELINE_BLOCKS_PER_CORE = 16
# R7's blocks per core. Its per-block fixed cost is two handshakes and one flush, while a smaller block
# smooths each core's DRAM write bursts against the reads and shortens fill/drain. Measured on cases
# 308 / 286 (us): 16 -> 1242 / 289, 64 -> 1144 / 275, 128 -> 1126 / 269, 256 -> 1132 / 273,
# 600 -> 1121 / 318 (286 then moves one tile per block).
SLOT_PIPELINE_BLOCKS_PER_CORE = 128

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
PACKER_UNTILIZE = "packer_untilize"
SHARDED_IN_PLACE = "sharded_in_place"
ROW_MAJOR_TO_TILE = "row_major_to_tile"
PACKER_TILIZE = "packer_tilize"
SAME_BANK_SLOT_WRITES = "same_bank_slot_writes"
_FACE_REGIMES = (FACE_DIRECT, SAME_BANK_SLOT_WRITES)
_UNTILIZE_REGIMES = (PACKER_UNTILIZE, SHARDED_IN_PLACE)
_COMPUTE_REGIMES = _UNTILIZE_REGIMES + (PACKER_TILIZE,)
_TO_TILE_REGIMES = (ROW_MAJOR_TO_TILE, PACKER_TILIZE)

CB_TILE_WINDOW = 0
CB_ROW_BLOCK = 1
CB_LEFT_FACES = 2
CB_NAN_MARKS = 3
CB_ROW_STAGE = 4
CB_TILE_BLOCK = 5
CB_SLOT_IMAGE = 6

# Per-tile record of a cb_nan_marks page (NanMarks in kernels/g6_to_layout_bfp8_nan_marks.hpp): a
# 64-bit bitmap of the tile's 64 bfp8 groups plus one 16-bit datum mask per group.
NAN_MARK_RECORD_BYTES = 8 + 2 * (4 * FACE_HW)
# cb_nan_marks pages are rounded up to this; it is never a NoC endpoint, so only RISC access alignment.
NAN_MARKS_PAGE_ALIGN = 16

# The R1 reader tags DRAM batches with trids 1..TILE_WINDOW_DEPTH and alternates two more over
# the reorder copies; ids are 4 bits and 0 means untagged.
_REORDER_TRIDS = 2
_MAX_TRID = 15
_MAX_TRID_OUTSTANDING = 255
assert TILE_WINDOW_DEPTH + _REORDER_TRIDS <= _MAX_TRID
# R5 tags its tile-row loads the same way; a split reorder adds the writer's two copy trids.
assert ROW_WINDOW_DEPTH + _REORDER_TRIDS * (2 if SPLIT_REORDER else 1) <= _MAX_TRID
assert not SPLIT_REORDER or ROW_WINDOW_DEPTH >= 2  # a pushed-ahead slot needs a second page
assert 2 * READ_BATCH_TILES <= _MAX_TRID_OUTSTANDING  # at most 2 reads per tile per batch
assert 0 <= READER_SLOT_EIGHTHS <= 8
assert 1 <= SLOT_READ_DEPTH <= SLOT_READ_DEPTH_MAX <= _MAX_TRID


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


def out_slot_bytes(geometry, out_alignment):
    """Aligned page slot of one ROW_MAJOR output row."""
    return _round_up(geometry.width * DATUM_BYTES, out_alignment)


def select_regime(geometry, dtype, sharded, to_tile, out_dtype, out_alignment):
    if to_tile:
        return PACKER_TILIZE if out_dtype == ttnn.bfloat8_b else ROW_MAJOR_TO_TILE
    if sharded:
        return SHARDED_IN_PLACE
    if dtype == ttnn.bfloat8_b:
        return PACKER_UNTILIZE
    if geometry.tensor_cols_tiles == 1 and geometry.width <= FACE_HW:
        if geometry.width * DATUM_BYTES < out_slot_bytes(geometry, out_alignment):
            return SAME_BANK_SLOT_WRITES
        return FACE_DIRECT
    return STAGED_REORDER


def _tile_row_stage_rows(regime, geometry):
    # The untilize packer emits, and the tilize unpacker consumes, a whole tile-row of 32 rows.
    return TILE_HW if regime in _COMPUTE_REGIMES else min(TILE_HW, geometry.height)


def _window_bytes():
    return TILE_WINDOW_DEPTH * READ_BATCH_TILES * TILE_BYTES


def _block_bytes(regime, block_rows_tiles, block_cols_tiles, stage_rows):
    if regime in _FACE_REGIMES:
        return block_rows_tiles * LEFT_SLOT_BYTES
    return block_rows_tiles * stage_rows * block_cols_tiles * TILE_ROW_BYTES


def _row_stage_tile_row_bytes(block_cols_tiles, stage_rows):
    return stage_rows * block_cols_tiles * TILE_ROW_BYTES


def _nan_marks_page_bytes(block_tiles):
    return _round_up(block_tiles * NAN_MARK_RECORD_BYTES, NAN_MARKS_PAGE_ALIGN)


def _slot_image_bytes(block_rows_tiles, stage_rows, slot_bytes):
    return block_rows_tiles * stage_rows * slot_bytes


def _max_block_tiles(regime, stage_rows, out_l1_share, in_tile_bytes, out_tile_bytes, slot_bytes):
    # The face regimes' reader tags a block's face reads (two per tile) with one trid, whose outstanding
    # count is capped.
    if regime == FACE_DIRECT:
        return min((L1_BUDGET_BYTES - out_l1_share) // (ROW_DEPTH * LEFT_SLOT_BYTES), _MAX_TRID_OUTSTANDING // 2)
    if regime == SAME_BANK_SLOT_WRITES:
        per_tile = SLOT_FACES_DEPTH * LEFT_SLOT_BYTES + SLOT_IMAGE_DEPTH * _slot_image_bytes(1, stage_rows, slot_bytes)
        return min((L1_BUDGET_BYTES - out_l1_share) // per_tile, _MAX_TRID_OUTSTANDING // 2)
    if regime == ROW_MAJOR_TO_TILE:
        # The row window scales with block_cols_tiles only; charging it per tile bounds it for any
        # block_rows_tiles >= 1.
        per_tile = ROW_WINDOW_DEPTH * stage_rows * TILE_ROW_BYTES + ROW_DEPTH * TILE_BYTES
        return (L1_BUDGET_BYTES - out_l1_share) // per_tile
    if regime == PACKER_TILIZE:
        return (L1_BUDGET_BYTES - out_l1_share) // (TILE_BLOCK_DEPTH * TILE_BYTES + ROW_DEPTH * out_tile_bytes)
    # The compute regimes' budget keeps the cb_nan_marks page rounding aside.
    marks_budget = L1_BUDGET_BYTES - out_l1_share - NAN_MARKS_DEPTH * NAN_MARKS_PAGE_ALIGN
    marks_per_tile = NAN_MARKS_DEPTH * NAN_MARK_RECORD_BYTES
    if regime == PACKER_UNTILIZE:
        return marks_budget // (TILE_BLOCK_DEPTH * in_tile_bytes + ROW_DEPTH * TILE_BYTES + marks_per_tile)
    if regime == SHARDED_IN_PLACE:
        # The input tile window is the caller's resident shard, outside the op's budget.
        return marks_budget // (ROW_DEPTH * TILE_BYTES + marks_per_tile)
    return (L1_BUDGET_BYTES - out_l1_share - _window_bytes()) // (ROW_DEPTH * stage_rows * TILE_ROW_BYTES)


def _choose_blocking(geometry, core_tiles_max, max_block_tiles, blocks_per_core):
    """(block_rows_tiles, block_cols_tiles): the whole row when it fits, else the core's column
    span; full-row blocks get tall enough for ~blocks_per_core blocks per core."""
    cols = geometry.tensor_cols_tiles
    block_cols_tiles = min(cols, core_tiles_max, max_block_tiles)
    if block_cols_tiles < cols:
        return 1, block_cols_tiles
    core_full_rows_max = core_tiles_max // cols
    rows = _div_up(core_full_rows_max, blocks_per_core)
    return max(1, min(rows, max_block_tiles // block_cols_tiles)), block_cols_tiles


@dataclass(frozen=True)
class ShardPlacement:
    """The input's shard placement, in tile units of the folded 2-D view (N * padded H, padded W)."""

    memory_layout: object
    grid: object  # ttnn.CoreRangeSet, exactly as the tensor's shard spec holds it
    shard_rows_tiles: int
    shard_cols_tiles: int
    row_major: bool

    @staticmethod
    def of(memory_config):
        spec = memory_config.shard_spec
        rows, cols = spec.shape
        assert rows % TILE_HW == 0 and cols % TILE_HW == 0, f"g6_to_layout: TILE shard shape {spec.shape}"
        return ShardPlacement(
            memory_config.memory_layout,
            spec.grid,
            rows // TILE_HW,
            cols // TILE_HW,
            spec.orientation == ttnn.ShardOrientation.ROW_MAJOR,
        )

    def core_rects(self, geometry):
        """[((x, y), (tile_row, rows, col, cols))]: every shard core and the live tile rectangle
        of its shard (rows or cols 0 for a shard wholly beyond the tensor). Mirrors the
        allocator's shard-to-core order: HEIGHT walks the grid's cores in orientation order; BLOCK
        puts shard (i, j) on core (x0 + j, y0 + i) (ROW_MAJOR) or (x0 + i, y0 + j) (COL_MAJOR)."""
        rects = []

        def live(begin, step, extent):
            return max(0, min(step, extent - begin))

        if self.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            assert self.shard_cols_tiles == geometry.tensor_cols_tiles
            cores = ttnn.corerange_to_cores(self.grid, None, self.row_major)
            for k, core in enumerate(cores):
                tile_row = k * self.shard_rows_tiles
                rows = live(tile_row, self.shard_rows_tiles, geometry.tensor_rows_tiles)
                rects.append(((core.x, core.y), (tile_row, rows, 0, geometry.tensor_cols_tiles)))
            return rects

        assert self.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        (box,) = self.grid.ranges()
        for core in ttnn.corerange_to_cores(self.grid, None, True):
            dx, dy = core.x - box.start.x, core.y - box.start.y
            i, j = (dy, dx) if self.row_major else (dx, dy)
            tile_row, col = i * self.shard_rows_tiles, j * self.shard_cols_tiles
            rows = live(tile_row, self.shard_rows_tiles, geometry.tensor_rows_tiles)
            cols = live(col, self.shard_cols_tiles, geometry.tensor_cols_tiles)
            if cols == 0:
                rows = 0
            rects.append(((core.x, core.y), (tile_row, rows, col, cols)))
        return rects


@dataclass
class Plan:
    regime: str
    geometry: Geometry
    grid: tuple
    num_cores: int  # cores with work
    all_cores: object
    # [((x, y), runtime args)]: (tile_begin, num_tiles) of the tile split, or for a shard
    # (tile_row, rows, col, cols) of its live rectangle.
    core_args: list
    block_rows_tiles: int
    block_cols_tiles: int
    tile_row_stage_rows: int
    in_tile_bytes: int
    out_tile_bytes: int
    out_slot_bytes: int  # aligned page slot of a ROW_MAJOR output row (None for a TILE output)
    cb_bytes: dict  # CB index -> total bytes the op allocates (a shard-backed CB allocates none)
    out_l1_share: int
    shard: ShardPlacement = None

    @property
    def nan_marks_page_bytes(self):
        return _nan_marks_page_bytes(self.block_rows_tiles * self.block_cols_tiles)

    @property
    def l1_bytes_per_core(self):
        return sum(self.cb_bytes.values()) + self.out_l1_share

    def report(self):
        if self.shard is None:
            cores = [
                {"core": core, "roles": self.roles, "tile_begin": begin, "num_tiles": count}
                for core, (begin, count) in self.core_args
            ]
            shard = None
        else:
            cores = [
                {"core": core, "roles": self.roles, "tile_row": r0, "rows": rows, "col": c0, "cols": cols}
                for core, (r0, rows, c0, cols) in self.core_args
            ]
            shard = {
                "memory_layout": self.shard.memory_layout,
                "grid": [((r.start.x, r.start.y), (r.end.x, r.end.y)) for r in self.shard.grid.ranges()],
                "shard_shape_tiles": (self.shard.shard_rows_tiles, self.shard.shard_cols_tiles),
                "orientation": "ROW_MAJOR" if self.shard.row_major else "COL_MAJOR",
            }
        return {
            "regime": self.regime,
            "num_cores": self.num_cores,
            "grid": self.grid,
            "shard": shard,
            "block_rows_tiles": self.block_rows_tiles,
            "block_cols_tiles": self.block_cols_tiles,
            "tile_row_stage_rows": self.tile_row_stage_rows,
            "read_batch_tiles": READ_BATCH_TILES if self.regime == STAGED_REORDER else None,
            "tile_window_depth": TILE_WINDOW_DEPTH if self.regime == STAGED_REORDER else None,
            "row_window_depth": ROW_WINDOW_DEPTH if self.regime == ROW_MAJOR_TO_TILE else None,
            "split_reorder": SPLIT_REORDER if self.regime == ROW_MAJOR_TO_TILE else None,
            "slot_image_depth": SLOT_IMAGE_DEPTH if self.regime == SAME_BANK_SLOT_WRITES else None,
            "slot_faces_depth": SLOT_FACES_DEPTH if self.regime == SAME_BANK_SLOT_WRITES else None,
            "slot_read_depth": SLOT_READ_DEPTH if self.regime == SAME_BANK_SLOT_WRITES else None,
            "slot_read_depth_max": SLOT_READ_DEPTH_MAX if self.regime == SAME_BANK_SLOT_WRITES else None,
            "slot_read_stagger": SLOT_READ_STAGGER if self.regime == SAME_BANK_SLOT_WRITES else None,
            "reader_slot_eighths": READER_SLOT_EIGHTHS if self.regime == SAME_BANK_SLOT_WRITES else None,
            "writer_slot_noc_copies": WRITER_SLOT_NOC_COPIES if self.regime == SAME_BANK_SLOT_WRITES else None,
            "out_slot_bytes": self.out_slot_bytes,
            "tile_block_depth": TILE_BLOCK_DEPTH if self.regime in (PACKER_UNTILIZE, PACKER_TILIZE) else None,
            "nan_marks_depth": (
                self.cb_bytes[CB_NAN_MARKS] // self.nan_marks_page_bytes if self.regime in _UNTILIZE_REGIMES else None
            ),
            "row_depth": ROW_DEPTH,
            "cb_bytes": dict(self.cb_bytes),
            "out_l1_share": self.out_l1_share,
            "l1_bytes_per_core": self.l1_bytes_per_core,
            "cores": cores,
        }

    @property
    def roles(self):
        return ("reader", "compute", "writer") if self.regime in _COMPUTE_REGIMES else ("reader", "writer")


def _tile_split(grid_x, grid_y, geometry):
    num_cores, all_cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(
        ttnn.CoreCoord(grid_x, grid_y), geometry.tensor_tiles, row_wise=True
    )
    core_args = []
    tile = 0
    for group, count in ((group_1, tiles_1), (group_2, tiles_2)):
        if count == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            core_args.append(((core.x, core.y), (tile, count)))
            tile += count
    assert tile == geometry.tensor_tiles, f"g6_to_layout: split covered {tile} of {geometry.tensor_tiles} tiles"
    return num_cores, all_cores, core_args, max(tiles_1, tiles_2)


def compute_plan(
    shape,
    grid,
    *,
    output_in_l1,
    num_l1_banks,
    out_alignment,
    dtype=ttnn.bfloat16,
    in_tile_bytes=TILE_BYTES,
    shard=None,
    to_tile=False,
    out_dtype=ttnn.bfloat16,
    out_tile_bytes=TILE_BYTES,
):
    """Regime, core assignment and block extents of one call. The regime depends on the shape, the
    input dtype and layout, the output dtype and alignment and whether the input is sharded, never
    on the grid.
    A TILE output (to_tile) is split over the same flattened tile space as a TILE input."""
    geometry = tensor_geometry(shape)
    regime = select_regime(geometry, dtype, shard is not None, to_tile, out_dtype, out_alignment)
    slot_bytes = None if to_tile else out_slot_bytes(geometry, out_alignment)
    stage_rows = _tile_row_stage_rows(regime, geometry)
    grid_x, grid_y = grid

    out_l1_share = 0
    if output_in_l1:
        if to_tile:
            out_pages, out_aligned_page_bytes = geometry.tensor_tiles, _round_up(out_tile_bytes, out_alignment)
        else:
            out_pages, out_aligned_page_bytes = geometry.out_pages, slot_bytes
        banks_per_core = max(1, _div_up(num_l1_banks, grid_x * grid_y))
        out_l1_share = _div_up(out_pages, num_l1_banks) * out_aligned_page_bytes * banks_per_core

    max_block_tiles = _max_block_tiles(regime, stage_rows, out_l1_share, in_tile_bytes, out_tile_bytes, slot_bytes)
    if max_block_tiles < 1:
        raise RuntimeError(
            f"g6_to_layout: an L1 output of {out_l1_share} B per core leaves no room for one block "
            f"within the {L1_BUDGET_BYTES} B L1 budget"
        )

    if shard is None:
        num_cores, all_cores, core_args, core_tiles_max = _tile_split(grid_x, grid_y, geometry)
        blocks_per_core = SLOT_PIPELINE_BLOCKS_PER_CORE if regime == SAME_BANK_SLOT_WRITES else PIPELINE_BLOCKS_PER_CORE
        block_rows_tiles, block_cols_tiles = _choose_blocking(
            geometry, core_tiles_max, max_block_tiles, blocks_per_core
        )
        blocks_per_core_max = None
    else:
        # The shard is the block: one pass over its live tile-rows unless L1 caps the block.
        all_cores = shard.grid
        core_args = shard.core_rects(geometry)
        num_cores = sum(1 for _, (_, rows, _, _) in core_args if rows > 0)
        block_cols_tiles = shard.shard_cols_tiles
        block_rows_tiles = min(shard.shard_rows_tiles, max_block_tiles // block_cols_tiles)
        if block_rows_tiles < 1:
            raise RuntimeError(
                f"g6_to_layout: one tile-row of a {block_cols_tiles}-tile shard does not fit the "
                f"{L1_BUDGET_BYTES} B L1 budget beside an L1 output of {out_l1_share} B per core"
            )
        blocks_per_core_max = _div_up(shard.shard_rows_tiles, block_rows_tiles)

    block_bytes = _block_bytes(regime, block_rows_tiles, block_cols_tiles, stage_rows)
    if regime == FACE_DIRECT:
        cb_bytes = {CB_LEFT_FACES: ROW_DEPTH * block_bytes}
    elif regime == SAME_BANK_SLOT_WRITES:
        cb_bytes = {
            CB_LEFT_FACES: SLOT_FACES_DEPTH * block_bytes,
            CB_SLOT_IMAGE: SLOT_IMAGE_DEPTH * _slot_image_bytes(block_rows_tiles, stage_rows, slot_bytes),
        }
    elif regime == STAGED_REORDER:
        cb_bytes = {CB_TILE_WINDOW: _window_bytes(), CB_ROW_BLOCK: ROW_DEPTH * block_bytes}
    elif regime == ROW_MAJOR_TO_TILE:
        cb_bytes = {
            CB_ROW_STAGE: ROW_WINDOW_DEPTH * _row_stage_tile_row_bytes(block_cols_tiles, stage_rows),
            CB_TILE_BLOCK: ROW_DEPTH * block_rows_tiles * block_cols_tiles * TILE_BYTES,
        }
    elif regime == PACKER_TILIZE:
        block_tiles = block_rows_tiles * block_cols_tiles
        cb_bytes = {
            CB_ROW_STAGE: TILE_BLOCK_DEPTH * block_tiles * TILE_BYTES,
            CB_TILE_BLOCK: ROW_DEPTH * block_tiles * out_tile_bytes,
        }
    elif regime == PACKER_UNTILIZE:
        block_tiles = block_rows_tiles * block_cols_tiles
        cb_bytes = {
            CB_TILE_WINDOW: TILE_BLOCK_DEPTH * block_tiles * in_tile_bytes,
            CB_ROW_BLOCK: ROW_DEPTH * block_bytes,
            CB_NAN_MARKS: NAN_MARKS_DEPTH * _nan_marks_page_bytes(block_tiles),
        }
    else:
        # A second row-block or marks page only helps when the shard takes more than one block.
        block_tiles = block_rows_tiles * block_cols_tiles
        cb_bytes = {
            CB_ROW_BLOCK: min(ROW_DEPTH, blocks_per_core_max) * block_bytes,
            CB_NAN_MARKS: min(NAN_MARKS_DEPTH, blocks_per_core_max) * _nan_marks_page_bytes(block_tiles),
        }

    return Plan(
        regime=regime,
        geometry=geometry,
        grid=(grid_x, grid_y),
        num_cores=num_cores,
        all_cores=all_cores,
        core_args=core_args,
        block_rows_tiles=block_rows_tiles,
        block_cols_tiles=block_cols_tiles,
        tile_row_stage_rows=stage_rows,
        in_tile_bytes=in_tile_bytes,
        out_tile_bytes=out_tile_bytes,
        out_slot_bytes=slot_bytes,
        cb_bytes=cb_bytes,
        out_l1_share=out_l1_share,
        shard=shard,
    )


def _cb(index, total_bytes, page_bytes, core_ranges, data_format=ttnn.bfloat16):
    return ttnn.CBDescriptor(
        total_size=total_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_bytes)],
    )


class G6ToLayoutProgram:
    """Program descriptor of one plan, built once. A later call with the same cache key only
    rebinds the two buffer addresses (common runtime arg 0 of the reader and of the writer) and,
    for a sharded input, the shard the tile window is backed on."""

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
        sharded = plan.regime == SHARDED_IN_PLACE
        self._sharded = sharded
        compute_ct_args = None
        compute_source = None

        if plan.regime in _FACE_REGIMES:
            cbs = [_cb(CB_LEFT_FACES, plan.cb_bytes[CB_LEFT_FACES], block_bytes, all_cores)]
            slot_writes = plan.regime == SAME_BANK_SLOT_WRITES
            if slot_writes:
                image_bytes = plan.cb_bytes[CB_SLOT_IMAGE]
                cbs.append(_cb(CB_SLOT_IMAGE, image_bytes, image_bytes // SLOT_IMAGE_DEPTH, all_cores))
            reader_source, writer_source = "g6_to_layout_face_reader.cpp", "g6_to_layout_face_writer.cpp"
            slot_ct_args = [CB_LEFT_FACES, int(slot_writes), CB_SLOT_IMAGE]
            read_knobs = (
                (SLOT_READ_DEPTH, SLOT_READ_DEPTH_MAX, SLOT_FACES_DEPTH, int(SLOT_READ_STAGGER))
                if slot_writes
                else (1, 1, ROW_DEPTH, 0)
            )
            reader_ct_args = geometry_ct_args + slot_ct_args + [READER_SLOT_EIGHTHS, plan.out_slot_bytes, *read_knobs]
            # The reader lays out the image by the output's bank interleaving, so it gets the output's
            # accessor args after its own.
            reader_out_ct_args = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
            writer_ct_args = (
                geometry_ct_args
                + slot_ct_args
                + [
                    SLOT_IMAGE_DEPTH,
                    plan.out_slot_bytes,
                    READER_SLOT_EIGHTHS,
                    int(WRITER_SLOT_NOC_COPIES),
                ]
            )
        elif plan.regime in _TO_TILE_REGIMES:
            reorder = plan.regime == ROW_MAJOR_TO_TILE
            stage_page_bytes = (
                _row_stage_tile_row_bytes(plan.block_cols_tiles, plan.tile_row_stage_rows) if reorder else TILE_BYTES
            )
            cbs = [
                _cb(CB_ROW_STAGE, plan.cb_bytes[CB_ROW_STAGE], stage_page_bytes, all_cores),
                _cb(CB_TILE_BLOCK, plan.cb_bytes[CB_TILE_BLOCK], plan.out_tile_bytes, all_cores, output_tensor.dtype),
            ]
            reader_source, writer_source = "g6_to_layout_row_reader.cpp", "g6_to_layout_tile_writer.cpp"
            split = int(reorder and SPLIT_REORDER)
            reader_ct_args = geometry_ct_args + [CB_ROW_STAGE, CB_TILE_BLOCK, int(reorder), ROW_WINDOW_DEPTH, split]
            writer_ct_args = geometry_ct_args + [CB_TILE_BLOCK, CB_ROW_STAGE, split, ROW_WINDOW_DEPTH]
            if not reorder:
                compute_source = "g6_to_layout_tilize_compute.cpp"
                compute_ct_args = geometry_ct_args + [CB_ROW_STAGE, CB_TILE_BLOCK]
        elif plan.regime == STAGED_REORDER:
            cbs = [
                _cb(CB_TILE_WINDOW, plan.cb_bytes[CB_TILE_WINDOW], TILE_BYTES, all_cores),
                _cb(CB_ROW_BLOCK, plan.cb_bytes[CB_ROW_BLOCK], block_bytes, all_cores),
            ]
            reader_source, writer_source = "g6_to_layout_staged_reader.cpp", "g6_to_layout_staged_writer.cpp"
            reader_ct_args = geometry_ct_args + [CB_TILE_WINDOW, CB_ROW_BLOCK, READ_BATCH_TILES, TILE_WINDOW_DEPTH]
            writer_ct_args = geometry_ct_args + [CB_ROW_BLOCK, 0, 0, 0, CB_NAN_MARKS]
        else:
            row_block = _cb(CB_ROW_BLOCK, plan.cb_bytes[CB_ROW_BLOCK], TILE_BYTES, all_cores)
            nan_marks = _cb(CB_NAN_MARKS, plan.cb_bytes[CB_NAN_MARKS], plan.nan_marks_page_bytes, all_cores)
            if sharded:
                cbs = [self._shard_window(input_tensor), row_block, nan_marks]
            else:
                window = _cb(
                    CB_TILE_WINDOW, plan.cb_bytes[CB_TILE_WINDOW], plan.in_tile_bytes, all_cores, input_tensor.dtype
                )
                cbs = [window, row_block, nan_marks]
            reader_source, writer_source = "g6_to_layout_tile_reader.cpp", "g6_to_layout_staged_writer.cpp"
            reader_ct_args = geometry_ct_args + [CB_TILE_WINDOW, int(sharded), CB_NAN_MARKS]
            writer_ct_args = geometry_ct_args + [CB_ROW_BLOCK, 1, int(sharded), 1, CB_NAN_MARKS]
            compute_source = "g6_to_layout_untilize_compute.cpp"
            compute_ct_args = geometry_ct_args + [CB_TILE_WINDOW, CB_ROW_BLOCK, int(sharded)]
        reader_ct_args += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        if plan.regime in _FACE_REGIMES:
            reader_ct_args += reader_out_ct_args
        writer_ct_args += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

        core_rt_args = ttnn.RuntimeArgs()
        for (x, y), args in plan.core_args:
            core_rt_args[x][y] = list(args)

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
        self._kernels = [self._reader, self._writer]
        if compute_ct_args is not None:
            # Fixed by the op's signature (no compute_kernel_config): a 16-bit DEST holds a bf16
            # value exactly, and the untilize / tilize does no arithmetic for fidelity to act on.
            self._kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / compute_source),
                    core_ranges=all_cores,
                    compile_time_args=compute_ct_args,
                    runtime_args=core_rt_args,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=False,
                        dst_full_sync_en=False,
                        math_approx_mode=False,
                    ),
                )
            )
        self._descriptor = ttnn.ProgramDescriptor(kernels=self._kernels, semaphores=[], cbs=cbs)

    @staticmethod
    def _shard_window(input_tensor):
        return ttnn.cb_descriptor_from_sharded_tensor(CB_TILE_WINDOW, input_tensor)

    def descriptor_for(self, input_tensor, output_tensor):
        self._reader.common_runtime_args = [input_tensor.buffer_address()]
        self._writer.common_runtime_args = [output_tensor.buffer_address()]
        self._descriptor.kernels = self._kernels
        if self._sharded:
            cbs = list(self._descriptor.cbs)
            cbs[0] = self._shard_window(input_tensor)
            self._descriptor.cbs = cbs
        return self._descriptor
