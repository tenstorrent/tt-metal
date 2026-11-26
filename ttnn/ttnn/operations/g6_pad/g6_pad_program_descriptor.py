# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_pad program descriptors for regimes E (extension, in place) and N (new storage).

Every per-call value (buffer addresses, fill bits) lives only in common runtime args, so a cached
descriptor is reused on every call of the same geometry; everything that changes the program
(geometry, segments, block size, core split) is a compile-time or per-core arg built once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FACE = 16

REGIME_EXTENSION = "E"
REGIME_NEW_STORAGE = "N"

# Model L1 budget per core for the op (BB-05).
L1_OP_BUDGET = 1347776
# Fewest work tiles a core is given before the grid stops growing (grid-synchronization knob).
MIN_CORE_WORK_TILES = 1
# Optional cap on N's block below the L1 clamp (overlap knob); None keeps the whole run per block.
N_BLOCK_TILES_TARGET = None
# NoC write alignment to DRAM and L1 on WH and BH.
NOC_WRITE_ALIGNMENT = 16
MAX_EXTENSION_SEGMENTS = 3

CB_WORK_TILES = 0
CB_FILL_TILE = 1


@dataclass
class CachedProgram:
    regime: str
    program: ttnn.ProgramDescriptor
    kernels: list
    # kernel index -> True if its common args start with the output address instead of x's.
    writes_output: list
    plan: dict = field(default_factory=dict)


def set_call_args(cached, x_addr, out_addr, fill_bits):
    """Write the per-call common runtime args and return the descriptor to dispatch."""
    for kernel, writes_output in zip(cached.kernels, cached.writes_output):
        kernel.common_runtime_args = [out_addr if writes_output else x_addr, fill_bits]
    # ProgramDescriptor.kernels is converted by value: reassign so the edits reach the descriptor.
    cached.program.kernels = cached.kernels
    return cached.program


def _prod(values):
    return math.prod(values) if values else 1


def _split_work(grid, num_work_tiles):
    """Contiguous runs of the folded work axis over a row-major prefix of the grid."""
    num_cores = min(grid.x * grid.y, -(-num_work_tiles // MIN_CORE_WORK_TILES))
    core_set = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        core_set, num_work_tiles, row_wise=True
    )
    runs = []
    begin = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            runs.append((core, begin, per_core))
            begin += per_core
    assert begin == num_work_tiles, f"g6_pad: work split covers {begin} of {num_work_tiles} tiles"
    return all_cores, runs, per_core_1


def _row_band_segments(row_begin, row_end, elem_bytes):
    """Byte ranges inside a tile covering rows [row_begin, row_end) x all 32 columns, adjacent
    face runs merged."""
    face_row_bytes = FACE * elem_bytes
    face_bytes = FACE * face_row_bytes
    segments = []
    for face in range(4):
        face_row_base = (face // 2) * FACE
        lo = max(row_begin, face_row_base)
        hi = min(row_end, face_row_base + FACE)
        if lo >= hi:
            continue
        start = face * face_bytes + (lo - face_row_base) * face_row_bytes
        end = face * face_bytes + (hi - face_row_base) * face_row_bytes
        if segments and segments[-1][1] == start:
            segments[-1][1] = end
        else:
            segments.append([start, end])
    return [(start, end - start) for start, end in segments]


def _plan(regime, num_work_tiles, all_cores, runs, block_tiles, buffer_depth, roles):
    return {
        "regime": regime,
        "num_work_tiles": num_work_tiles,
        "num_cores": len(runs),
        "block_tiles": block_tiles,
        "buffer_depth": buffer_depth,
        "roles": {role: str(all_cores) for role in roles},
    }


def _extension_program(input_tensor, highs, grid):
    shape = list(input_tensor.shape)
    elem_bytes = input_tensor.element_size()
    tile_bytes = ttnn.tile_size(input_tensor.dtype)

    out_tensor_slices = _prod(shape[:-2])
    out_tensor_row_tiles = -(-shape[-2] // TILE)
    out_tensor_col_tiles = -(-shape[-1] // TILE)
    in_row_rem = shape[-2] % TILE
    out_row_end = (shape[-2] + highs[-2]) % TILE or TILE
    assert in_row_rem != 0 and in_row_rem < out_row_end

    segments = _row_band_segments(in_row_rem, out_row_end, elem_bytes)
    assert 1 <= len(segments) <= MAX_EXTENSION_SEGMENTS, segments
    for offset, length in segments:
        assert offset % NOC_WRITE_ALIGNMENT == 0 and length % NOC_WRITE_ALIGNMENT == 0, segments
    segment_args = []
    for k in range(MAX_EXTENSION_SEGMENTS):
        segment_args += list(segments[k]) if k < len(segments) else [0, 0]

    num_work_tiles = out_tensor_slices * out_tensor_col_tiles
    all_cores, runs, max_core_work_tiles = _split_work(grid, num_work_tiles)
    block_tiles = max_core_work_tiles

    ct_args = [
        CB_FILL_TILE,
        elem_bytes,
        tile_bytes,
        len(segments),
        *segment_args,
        out_tensor_row_tiles,
        out_tensor_col_tiles,
        block_tiles,
    ]
    ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    rt_args = ttnn.RuntimeArgs()
    for core, begin, tiles in runs:
        rt_args[core.x][core.y] = [begin, tiles, -(-tiles // block_tiles)]

    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_extension_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        common_runtime_args=[0, 0],
        config=ttnn.WriterConfigDescriptor(),
    )
    cb_fill_tile = ttnn.CBDescriptor(
        total_size=tile_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_FILL_TILE, data_format=input_tensor.dtype, page_size=tile_bytes)
        ],
    )
    kernels = [writer]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_fill_tile])
    plan = _plan(REGIME_EXTENSION, num_work_tiles, all_cores, runs, block_tiles, 1, ["extension_writer"])
    plan["segments"] = segments
    return CachedProgram(REGIME_EXTENSION, program, kernels, [False], plan)


def _leading2(dims):
    """Leading dims padded on the left to exactly two (rank 2..4)."""
    lead = list(dims)
    assert len(lead) <= 2
    return [1] * (2 - len(lead)) + lead


def _new_storage_program(input_tensor, output_tensor, highs, grid):
    in_shape = list(input_tensor.shape)
    out_shape = [d + h for d, h in zip(in_shape, highs)]
    elem_bytes = input_tensor.element_size()
    tile_bytes = ttnn.tile_size(input_tensor.dtype)

    out_lead = _leading2(out_shape[:-2])
    in_lead = _leading2(in_shape[:-2])
    out_tensor_slices = _prod(out_lead)
    out_tensor_row_tiles = -(-out_shape[-2] // TILE)
    out_tensor_col_tiles = -(-out_shape[-1] // TILE)
    in_tensor_row_tiles = -(-in_shape[-2] // TILE)
    in_tensor_col_tiles = -(-in_shape[-1] // TILE)
    in_row_rem = in_shape[-2] % TILE
    in_col_rem = in_shape[-1] % TILE
    row_mask_on = int(in_row_rem != 0 and out_shape[-2] > in_shape[-2])
    col_mask_on = int(in_col_rem != 0 and out_shape[-1] > in_shape[-1])

    num_work_tiles = out_tensor_slices * out_tensor_row_tiles * out_tensor_col_tiles
    all_cores, runs, max_core_work_tiles = _split_work(grid, num_work_tiles)

    block_tiles_cap = (L1_OP_BUDGET - tile_bytes) // (2 * tile_bytes)
    block_tiles = min(max_core_work_tiles, block_tiles_cap)
    if N_BLOCK_TILES_TARGET is not None:
        block_tiles = min(block_tiles, N_BLOCK_TILES_TARGET)
    buffer_depth = 2 if -(-max_core_work_tiles // block_tiles) > 1 else 1

    def has_input(w):
        tc = w % out_tensor_col_tiles
        tr = (w // out_tensor_col_tiles) % out_tensor_row_tiles
        s = w // (out_tensor_row_tiles * out_tensor_col_tiles)
        l0, l1 = divmod(s, out_lead[1])
        return l0 < in_lead[0] and l1 < in_lead[1] and tr < in_tensor_row_tiles and tc < in_tensor_col_tiles

    geometry = [
        out_lead[1],
        in_lead[0],
        in_lead[1],
        out_tensor_row_tiles,
        out_tensor_col_tiles,
        in_tensor_row_tiles,
        in_tensor_col_tiles,
    ]

    reader_ct = [CB_WORK_TILES, elem_bytes, tile_bytes, block_tiles, *geometry, in_row_rem, in_col_rem]
    reader_ct += [row_mask_on, col_mask_on]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct = [CB_WORK_TILES, CB_FILL_TILE, elem_bytes, tile_bytes, block_tiles, *geometry]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    has_fill_only_tiles = False
    fill_only_per_core = []
    for core, begin, tiles in runs:
        fill_only = sum(1 for w in range(begin, begin + tiles) if not has_input(w))
        fill_only_per_core.append(fill_only)
        has_fill_only_tiles |= fill_only > 0
        num_blocks = -(-tiles // block_tiles)
        reader_rt[core.x][core.y] = [begin, tiles, num_blocks]
        writer_rt[core.x][core.y] = [begin, tiles, num_blocks, int(fill_only > 0)]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        common_runtime_args=[0, 0],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_pad_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        common_runtime_args=[0, 0],
        config=ttnn.WriterConfigDescriptor(),
    )

    cbs = [
        ttnn.CBDescriptor(
            total_size=block_tiles * buffer_depth * tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_WORK_TILES, data_format=input_tensor.dtype, page_size=tile_bytes
                )
            ],
        )
    ]
    if has_fill_only_tiles:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_FILL_TILE, data_format=input_tensor.dtype, page_size=tile_bytes
                    )
                ],
            )
        )
    assert sum(cb.total_size for cb in cbs) <= L1_OP_BUDGET

    kernels = [reader, writer]
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    plan = _plan(
        REGIME_NEW_STORAGE,
        num_work_tiles,
        all_cores,
        runs,
        block_tiles,
        buffer_depth,
        ["new_storage_reader", "new_storage_writer"],
    )
    plan["fill_only_tiles"] = sum(fill_only_per_core)
    return CachedProgram(REGIME_NEW_STORAGE, program, kernels, [False, True], plan)


def create_program_descriptor(regime, input_tensor, output_tensor, highs, grid):
    if regime == REGIME_EXTENSION:
        return _extension_program(input_tensor, highs, grid)
    return _new_storage_program(input_tensor, output_tensor, highs, grid)
