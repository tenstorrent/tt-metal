# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_expand program descriptor — regime R1 (op_design.md, Blocking Model).

Knobs (each defined once here; every dependent quantity derives from them):
  grid          — core grid fed to split_work_to_cores
  COL_BLOCK_CAP — L1 clamp on col_block (broadcast tiles resident per fetch block)
  col_block     — min(max_cols_per_core, COL_BLOCK_CAP), one CT arg to the reader
  bcast_depth   — blocks of col_block pages in cb_bcast_tiles (1 when one block per core)
  row_fetch_bytes — max(32, DRAM alignment): bytes fetched at offsets 0 and FACE_BYTES of a page

The descriptor is cached per geometry (BB-07): a repeat call only refreshes the two
buffer addresses, which are common runtime args.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_H = 32
TILE_W = 32
FACE_ROWS = 16
COL_BLOCK_CAP = 64  # tiles (128 KiB of bf16) — L1 clamp on col_block
REGIME = "R1-colmajor-row0fetch"

CB_BCAST_TILES = 0  # broadcast tiles: row 0 fetched + replicated in place; also the write source

_cache = {}


def _div_up(a, b):
    return (a + b - 1) // b


def _ordered_cores(core_group):
    return list(ttnn.corerange_to_cores(core_group, row_wise=True))


def _build(input_tensor, output_tensor, grid):
    in_shape = list(input_tensor.shape)
    out_shape = list(output_tensor.shape)
    n_img = out_shape[0] * out_shape[1]
    Wt = _div_up(out_shape[3], TILE_W)
    Ht = _div_up(out_shape[2], TILE_H)
    num_cols = n_img * Wt
    total_tiles = num_cols * Ht
    assert input_tensor.buffer_num_pages() == num_cols, (input_tensor.buffer_num_pages(), num_cols, in_shape)
    assert output_tensor.buffer_num_pages() == total_tiles

    page_size = input_tensor.buffer_page_size()
    assert output_tensor.buffer_page_size() == page_size
    elem_size = input_tensor.element_size()
    face_bytes = FACE_ROWS * FACE_ROWS * elem_size  # 512 B for bf16
    face_row_bytes = FACE_ROWS * elem_size  # 32 B for bf16
    row_fetch_bytes = max(face_row_bytes, ttnn.get_dram_alignment())
    assert row_fetch_bytes <= face_bytes

    # ---- Work split: column-major linear order over output tiles ----
    (num_cores, all_cores, core_group_1, core_group_2, n_g1, n_g2) = ttnn.split_work_to_cores(
        grid, total_tiles, row_wise=True
    )
    assignments = []  # (core, start_t, n_t)
    start_t = 0
    for group, n_t in ((core_group_1, n_g1), (core_group_2, n_g2)):
        if n_t == 0:
            continue
        for core in _ordered_cores(group):
            assignments.append((core, start_t, n_t))
            start_t += n_t
    assert start_t == total_tiles, (start_t, total_tiles)

    max_cols_per_core = max((s + n - 1) // Ht - s // Ht + 1 for _, s, n in assignments)
    col_block = min(max_cols_per_core, COL_BLOCK_CAP)
    num_col_blocks_max = _div_up(max_cols_per_core, col_block)
    bcast_depth = 1 if num_col_blocks_max == 1 else 2

    # ---- CB ----
    cb_bcast_tiles = ttnn.CBDescriptor(
        total_size=bcast_depth * col_block * page_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BCAST_TILES,
                data_format=input_tensor.dtype,
                page_size=page_size,
            )
        ],
    )

    # ---- Kernels ----
    reader_ct_args = [CB_BCAST_TILES, Ht, col_block, page_size, row_fetch_bytes, face_bytes, face_row_bytes]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args = [CB_BCAST_TILES, Ht, Wt, page_size]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    for core, s, n in assignments:
        reader_rt[core.x][core.y] = [s, n]
        writer_rt[core.x][core.y] = [s, n]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        common_runtime_args=[input_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel],
        semaphores=[],
        cbs=[cb_bcast_tiles],
    )
    plan = {
        "regime": REGIME,
        "num_cores": num_cores,
        "total_tiles": total_tiles,
        "Ht": Ht,
        "Wt": Wt,
        "col_block": col_block,
        "bcast_depth": bcast_depth,
        "row_fetch_bytes": row_fetch_bytes,
        "max_cols_per_core": max_cols_per_core,
    }
    return descriptor, plan


def get_program(input_tensor, output_tensor):
    """Return (ProgramDescriptor, plan); built once per geometry, addresses refreshed per call."""
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    key = (
        device.id(),
        tuple(input_tensor.shape),
        tuple(output_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        str(output_tensor.memory_config()),
        (grid.x, grid.y),
    )
    entry = _cache.get(key)
    if entry is None:
        entry = _build(input_tensor, output_tensor, grid)
        _cache[key] = entry
    descriptor, plan = entry
    kernels = descriptor.kernels
    kernels[0].common_runtime_args = [input_tensor.buffer_address()]
    kernels[1].common_runtime_args = [output_tensor.buffer_address()]
    descriptor.kernels = kernels
    return descriptor, plan
