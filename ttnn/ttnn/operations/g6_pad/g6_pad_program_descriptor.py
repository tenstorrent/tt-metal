# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_pad program descriptors: regime E (in-place extension) and regime N (new storage).

Both regimes fold the output tile index space (slice, tile row, tile column) into one work axis
`w`, tile column fastest, and split it over the grid in contiguous runs. A descriptor depends only
on the call's geometry; the per-call values (buffer addresses, fill bits) are common runtime args
written by the entry point on every call, so a cached descriptor is reused as is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FACE = 16

# L1 the model leaves to the op on every core (BB-05).
L1_OP_BUDGET = 1347776

# NoC write alignment to DRAM and L1 on both Wormhole and Blackhole.
NOC_WRITE_ALIGNMENT = 16

# ---- Blocking knobs (single source each; every derived quantity is computed from them) ----
# Fewest work tiles a core is given before another core is added (grid-synchronization lamp).
MIN_CORE_WORK_TILES = 1
# Extra clamp on the N block (overlap lamp); None = only the L1 clamp applies.
N_BLOCK_TILES_LIMIT = None
# N CB depth once a core streams more than one block.
N_MULTI_BLOCK_BUFFER_DEPTH = 2
# Most E write segments per touched tile (rows [a, b) of a tile split by faces, merged).
MAX_EXTENSION_SEGMENTS = 3
# Share (numerator, denominator) of each core's E run written by NCRISC on NoC0; BRISC writes the
# rest on NoC1. Two NoCs give the writes two disjoint link sets. (0, 1) = BRISC only.
E_SECOND_RISC_SHARE = (2, 5)
# E: start each writer's walk on DRAM bank (writer ordinal % num_banks) instead of all writers
# starting on the banks their run offsets happen to hit. Off: p2-p4 are bound by aggregate DRAM
# write bandwidth at 110 cores x 2 RISCs, where the stagger measured flat.
BANK_STAGGER = False

CB_WORK_TILES = 0
CB_FILL_TILE = 1
CB_FILL_TILE_SECOND_RISC = 2


def knobs_key():
    return (MIN_CORE_WORK_TILES, N_BLOCK_TILES_LIMIT, N_MULTI_BLOCK_BUFFER_DEPTH, E_SECOND_RISC_SHARE, BANK_STAGGER)


def _ceil_div(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Geometry:
    """Per-image tile geometry of one call (ceil per dim, never a flattened floor)."""

    in_shape: tuple
    out_shape: tuple
    elem_bytes: int
    tile_bytes: int
    in_lead: tuple  # (d_0, d_1) of the leading dims, padded on the left with 1 to two entries
    out_lead: tuple
    in_row_tiles: int
    out_row_tiles: int
    in_col_tiles: int
    out_col_tiles: int
    in_row_rem: int
    in_col_rem: int
    out_row_end: int

    @property
    def in_tensor_slices(self):
        return self.in_lead[0] * self.in_lead[1]

    @property
    def out_tensor_slices(self):
        return self.out_lead[0] * self.out_lead[1]

    @property
    def row_mask_rem(self):
        """Rows of the input's last tile row that hold data; 0 when no row mask is needed (the
        input's padding rows are output padding too when dim -2 does not grow)."""
        return self.in_row_rem if self.out_shape[-2] > self.in_shape[-2] else 0

    @property
    def col_mask_rem(self):
        return self.in_col_rem if self.out_shape[-1] > self.in_shape[-1] else 0


def make_geometry(in_shape, out_shape, elem_bytes, tile_bytes):
    in_shape, out_shape = tuple(in_shape), tuple(out_shape)
    rank = len(in_shape)
    if not 2 <= rank <= 4:
        raise ValueError(f"g6_pad: rank {rank} outside 2..4")

    def lead(shape):
        dims = list(shape[:-2])
        return tuple([1] * (2 - len(dims)) + dims)

    out_row_end = out_shape[-2] % TILE or TILE
    return Geometry(
        in_shape=in_shape,
        out_shape=out_shape,
        elem_bytes=elem_bytes,
        tile_bytes=tile_bytes,
        in_lead=lead(in_shape),
        out_lead=lead(out_shape),
        in_row_tiles=_ceil_div(in_shape[-2], TILE),
        out_row_tiles=_ceil_div(out_shape[-2], TILE),
        in_col_tiles=_ceil_div(in_shape[-1], TILE),
        out_col_tiles=_ceil_div(out_shape[-1], TILE),
        in_row_rem=in_shape[-2] % TILE,
        in_col_rem=in_shape[-1] % TILE,
        out_row_end=out_row_end,
    )


def row_band_segments(first_row, end_row, elem_bytes):
    """Byte ranges (offset, length) inside one tile covering tile rows [first_row, end_row) across
    all 32 columns. Faces are 16x16 in order face0 (rows 0-15, cols 0-15), face1 (rows 0-15,
    cols 16-31), face2, face3; adjacent ranges are merged."""
    face_row_bytes = FACE * elem_bytes
    face_bytes = FACE * face_row_bytes
    ranges = []
    for face in range(4):
        face_first_row = 0 if face < 2 else FACE
        lo = max(first_row, face_first_row)
        hi = min(end_row, face_first_row + FACE)
        if lo >= hi:
            continue
        start = face * face_bytes + (lo - face_first_row) * face_row_bytes
        end = face * face_bytes + (hi - face_first_row) * face_row_bytes
        if ranges and ranges[-1][1] == start:
            ranges[-1][1] = end
        else:
            ranges.append([start, end])
    return [(start, end - start) for start, end in ranges]


def _split_work(grid, num_work_tiles):
    """Contiguous runs of `w` over a row-major prefix of the grid.

    Returns (all_cores, [(core, core_work_begin, core_work_tiles)], max_core_work_tiles,
    [(group core range set, core_work_tiles)])."""
    num_cores_cap = min(grid.x * grid.y, _ceil_div(num_work_tiles, MIN_CORE_WORK_TILES))
    candidate_cores = ttnn.num_cores_to_corerangeset(num_cores_cap, grid, True)
    _, all_cores, core_group_1, core_group_2, group_1_tiles, group_2_tiles = ttnn.split_work_to_cores(
        candidate_cores, num_work_tiles, True
    )
    groups = [(g, n) for g, n in ((core_group_1, group_1_tiles), (core_group_2, group_2_tiles)) if n > 0]
    assignment = []
    core_work_begin = 0
    for group, core_work_tiles in groups:
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, core_work_begin, core_work_tiles))
            core_work_begin += core_work_tiles
    assert core_work_begin == num_work_tiles, (core_work_begin, num_work_tiles)
    return all_cores, assignment, group_1_tiles, groups


def _cb(index, num_pages, geom, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * geom.tile_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=geom.tile_bytes)],
    )


# ------------------------------------------------------------------------------------------------
# Regime E: write-only new rows of the last tile row of each slice, in place
# ------------------------------------------------------------------------------------------------


def _second_risc_tiles(core_work_tiles):
    numerator, denominator = E_SECOND_RISC_SHARE
    return core_work_tiles * numerator // denominator


def _extension_page(geom, w):
    slice_index, tile_col = divmod(w, geom.in_col_tiles)
    return (slice_index * geom.in_row_tiles + geom.in_row_tiles - 1) * geom.in_col_tiles + tile_col


def _walk_offset(geom, run_begin, run_tiles, target_bank, num_banks):
    """Offset into [run_begin, run_begin + run_tiles) whose page sits on target_bank (interleaved
    page p lives on bank p % num_banks); 0 when stagger is off or no page of the run matches."""
    if not BANK_STAGGER:
        return 0
    for offset in range(min(run_tiles, num_banks)):
        if _extension_page(geom, run_begin + offset) % num_banks == target_bank:
            return offset
    return 0


def build_extension_descriptor(geom, input_tensor, grid):
    """Returns (descriptor, plan, {role: kernel index}). Common RT args of every E writer:
    [x_addr, fill_bits].

    Each core's run is split in two contiguous parts: BRISC (NoC1) writes the first, NCRISC
    (NoC0) the second, each from its own fill page. Both run the same kernel source."""
    num_work_tiles = geom.out_tensor_slices * geom.out_col_tiles
    all_cores, assignment, max_core_work_tiles, groups = _split_work(grid, num_work_tiles)

    segments = row_band_segments(geom.in_row_rem, geom.out_row_end, geom.elem_bytes)
    assert 1 <= len(segments) <= MAX_EXTENSION_SEGMENTS, segments
    for offset, length in segments:
        assert offset % NOC_WRITE_ALIGNMENT == 0 and length % NOC_WRITE_ALIGNMENT == 0, segments
    padded_segments = list(segments) + [(0, 0)] * (MAX_EXTENSION_SEGMENTS - len(segments))
    accessor_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()

    def writer_kernel(cb_fill_index, block_tiles, core_ranges, rt_args, config):
        ct_args = [
            cb_fill_index,
            geom.tile_bytes,
            geom.elem_bytes,
            geom.in_row_tiles,
            geom.in_col_tiles,
            block_tiles,
            len(segments),
        ]
        for offset, length in padded_segments:
            ct_args += [offset, length]
        ct_args += accessor_ct_args
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "g6_pad_extension_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=ct_args,
            runtime_args=rt_args,
            common_runtime_args=[0, 0],
            config=config,
        )

    # BRISC part: [begin, begin + n - second); NCRISC part: the remaining `second` tiles.
    first_block_tiles = max_core_work_tiles - _second_risc_tiles(max_core_work_tiles)
    second_block_tiles = _second_risc_tiles(max_core_work_tiles)
    num_banks = ttnn.device.get_memory_view(input_tensor.device(), input_tensor.memory_config().buffer_type).num_banks
    first_rt_args = ttnn.RuntimeArgs()
    second_rt_args = ttnn.RuntimeArgs()
    for ordinal, (core, core_work_begin, core_work_tiles) in enumerate(assignment):
        second_tiles = _second_risc_tiles(core_work_tiles)
        first_tiles = core_work_tiles - second_tiles
        first_rt_args[core.x][core.y] = [
            core_work_begin,
            first_tiles,
            _ceil_div(first_tiles, first_block_tiles),
            _walk_offset(geom, core_work_begin, first_tiles, ordinal % num_banks, num_banks),
        ]
        if second_tiles > 0:
            second_begin = core_work_begin + first_tiles
            second_rt_args[core.x][core.y] = [
                second_begin,
                second_tiles,
                _ceil_div(second_tiles, second_block_tiles),
                _walk_offset(geom, second_begin, second_tiles, (ordinal + num_banks // 2) % num_banks, num_banks),
            ]

    kernels = [writer_kernel(CB_FILL_TILE, first_block_tiles, all_cores, first_rt_args, ttnn.WriterConfigDescriptor())]
    cbs = [_cb(CB_FILL_TILE, 1, geom, input_tensor.dtype, all_cores)]
    roles = {"extension_writer": 0}
    plan_roles = {str(all_cores): "extension_writer (BRISC, NoC1)"}
    second_groups = [group for group, core_work_tiles in groups if _second_risc_tiles(core_work_tiles) > 0]
    if second_groups:
        second_cores = second_groups[0]
        for group in second_groups[1:]:
            second_cores = second_cores.merge(group)
        kernels.append(
            writer_kernel(
                CB_FILL_TILE_SECOND_RISC,
                second_block_tiles,
                second_cores,
                second_rt_args,
                ttnn.ReaderConfigDescriptor(),
            )
        )
        cbs.append(_cb(CB_FILL_TILE_SECOND_RISC, 1, geom, input_tensor.dtype, second_cores))
        roles["extension_writer_second_risc"] = 1
        plan_roles[str(second_cores)] = "extension_writer (NCRISC, NoC0)"

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    plan = {
        "regime": "E",
        "num_work_tiles": num_work_tiles,
        "num_cores": len(assignment),
        "block_tiles": max_core_work_tiles,
        "block_tiles_per_risc": (first_block_tiles, second_block_tiles),
        "buffer_depth": None,
        "segments": segments,
        "roles": plan_roles,
    }
    return descriptor, plan, roles


# ------------------------------------------------------------------------------------------------
# Regime N: new storage, one pass, boundary tiles assembled in L1
# ------------------------------------------------------------------------------------------------


def _slice_exists(geom, out_slice):
    lead_0, lead_1 = divmod(out_slice, geom.out_lead[1])
    return lead_0 < geom.in_lead[0] and lead_1 < geom.in_lead[1]


def count_input_backed(geom, w_end):
    """Number of input-backed output tiles among w in [0, w_end)."""
    slice_tiles = geom.out_row_tiles * geom.out_col_tiles
    full_slices, rem = divmod(w_end, slice_tiles)
    lead_0_full, lead_1_rem = divmod(full_slices, geom.out_lead[1])
    existing_slices = min(lead_0_full, geom.in_lead[0]) * geom.in_lead[1]
    if lead_0_full < geom.in_lead[0]:
        existing_slices += min(lead_1_rem, geom.in_lead[1])
    count = existing_slices * geom.in_row_tiles * geom.in_col_tiles
    if rem and _slice_exists(geom, full_slices):
        tile_row, tile_col = divmod(rem, geom.out_col_tiles)
        count += min(tile_row, geom.in_row_tiles) * geom.in_col_tiles
        if tile_row < geom.in_row_tiles:
            count += min(tile_col, geom.in_col_tiles)
    return count


def geometry_ct_args(geom):
    """The geometry block shared by the N reader and writer (g6_pad_new_storage_walk.hpp)."""
    return [
        geom.out_lead[0],
        geom.out_lead[1],
        geom.in_lead[0],
        geom.in_lead[1],
        geom.out_row_tiles,
        geom.out_col_tiles,
        geom.in_row_tiles,
        geom.in_col_tiles,
        geom.row_mask_rem,
        geom.col_mask_rem,
    ]


def new_storage_block(geom, max_core_work_tiles):
    block_tiles_cap = (L1_OP_BUDGET - geom.tile_bytes) // (2 * geom.tile_bytes)
    block_tiles = min(max_core_work_tiles, block_tiles_cap)
    if N_BLOCK_TILES_LIMIT is not None:
        block_tiles = min(block_tiles, N_BLOCK_TILES_LIMIT)
    buffer_depth = N_MULTI_BLOCK_BUFFER_DEPTH if _ceil_div(max_core_work_tiles, block_tiles) > 1 else 1
    return block_tiles, buffer_depth


def build_new_storage_descriptor(geom, input_tensor, output_tensor, grid):
    """Returns (descriptor, plan, {role: kernel index}). Common RT args: reader [x_addr, fill_bits],
    writer [out_addr, fill_bits]."""
    num_work_tiles = geom.out_tensor_slices * geom.out_row_tiles * geom.out_col_tiles
    all_cores, assignment, max_core_work_tiles, _ = _split_work(grid, num_work_tiles)
    block_tiles, buffer_depth = new_storage_block(geom, max_core_work_tiles)
    has_fill_only_tiles = count_input_backed(geom, num_work_tiles) < num_work_tiles

    geometry_args = geometry_ct_args(geom)
    reader_ct_args = [CB_WORK_TILES, geom.tile_bytes, geom.elem_bytes, block_tiles] + geometry_args
    reader_ct_args += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_ct_args = [
        CB_WORK_TILES,
        CB_FILL_TILE,
        geom.tile_bytes,
        geom.elem_bytes,
        block_tiles,
        int(has_fill_only_tiles),
    ] + geometry_args
    writer_ct_args += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    for core, core_work_begin, core_work_tiles in assignment:
        num_blocks_this_core = _ceil_div(core_work_tiles, block_tiles)
        core_input_tiles = count_input_backed(geom, core_work_begin + core_work_tiles) - count_input_backed(
            geom, core_work_begin
        )
        core_has_fill_only = int(core_input_tiles < core_work_tiles)
        reader_rt_args[core.x][core.y] = [core_work_begin, core_work_tiles, num_blocks_this_core]
        writer_rt_args[core.x][core.y] = [core_work_begin, core_work_tiles, num_blocks_this_core, core_has_fill_only]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        common_runtime_args=[0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        common_runtime_args=[0, 0],
        config=ttnn.WriterConfigDescriptor(),
    )
    cbs = [_cb(CB_WORK_TILES, block_tiles * buffer_depth, geom, input_tensor.dtype, all_cores)]
    if has_fill_only_tiles:
        cbs.append(_cb(CB_FILL_TILE, 1, geom, input_tensor.dtype, all_cores))
    footprint = geom.tile_bytes * (block_tiles * buffer_depth + int(has_fill_only_tiles))
    assert footprint <= L1_OP_BUDGET, footprint

    descriptor = ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=cbs)
    plan = {
        "regime": "N",
        "num_work_tiles": num_work_tiles,
        "num_cores": len(assignment),
        "block_tiles": block_tiles,
        "buffer_depth": buffer_depth,
        "has_fill_only_tiles": has_fill_only_tiles,
        "l1_bytes_per_core": footprint,
        "roles": {str(all_cores): "new_storage_reader+new_storage_writer"},
    }
    return descriptor, plan, {"new_storage_reader": 0, "new_storage_writer": 1}


__all__ = [
    "Geometry",
    "make_geometry",
    "row_band_segments",
    "count_input_backed",
    "build_extension_descriptor",
    "build_new_storage_descriptor",
    "knobs_key",
]
