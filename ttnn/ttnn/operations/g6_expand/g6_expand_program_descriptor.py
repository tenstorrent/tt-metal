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

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_H = 32
TILE_W = 32
FACE_ROWS = 16
COL_BLOCK_CAP = 64  # tiles (128 KiB of bf16) — L1 clamp on col_block
REGIME = "R1-colmajor-row0fetch"
# Replication mechanism (perf lamp): rows per face the reader RISC-V builds before local NoC
# doubling copies take over; 2 * FACE_ROWS = all rows by RISC-V stores (no NoC copies).
REPLICATE_CPU_ROWS = 16
# Core placement (perf lamp): range k of the column-major order goes to core
# ordered[(k * CORE_ORDER_STRIDE) % num_cores] (stride made coprime with num_cores). 1 = the
# split's own row-wise order; a larger stride spreads the cores of one column (= one DRAM bank
# when Wt % num_banks == 0) over the grid instead of clustering them on one NoC row.
CORE_ORDER_STRIDE = 1
# Write issue (perf lamp): the reader (NCRISC, NoC0) writes the last
# (seg_len * READER_WRITE_EIGHTHS) // 8 tiles of every column segment; the writer the rest.
# 0 = all writes on the writer (BRISC, NoC1).
READER_WRITE_EIGHTHS = 4
# Grid size (perf lamp "grid synchronization"): the split uses
#   target = min(grid cores, MAX_CORES, ceil(total_tiles / MIN_TILES_PER_CORE))
# cores, taken row-wise (full NoC rows first) from the device grid. None / 1 = whole grid.
MAX_CORES = 80  # measured on W=512 (Refinement 1): 80 → ~2.31 µs vs 110 → ~2.49 µs
MIN_TILES_PER_CORE = 1
# Broadcast-tile build (perf lamp "replicate on the critical path", Refinement 2): an output tile
# is [F0 F1 F0 F1]. 1 = build faces 0/1 only in a 1 KiB slot and write every output page as two
# 1 KiB writes from it (no faces 0/1 -> 2/3 copy, half the slot); 0 = build the whole 2 KiB tile
# and write it as one page.
HALF_TILE = 1
# Row-0 fetch requests per column (design lamp "row-fetch request count"): 2 = row_fetch_bytes at
# offsets 0 and FACE_BYTES; 1 = one read of FACE_BYTES + row_fetch_bytes from offset 0.
ROW_FETCH_REQUESTS = 2
# Replicate NoC topology (with REPLICATE_CPU_ROWS < FACE_ROWS): 0 = doubling (one barrier per
# level); 1 = fan-out (each remaining chunk read from the CPU-built prefix, one barrier total).
REPLICATE_NOC_FANOUT = 0
# Replicate split across the two data-movement RISCs: 1 = the writer (BRISC) fetches + builds
# face 1 (3) while the reader (NCRISC) builds face 0 (2), local semaphore handoff. Applies only
# when the build is pure RISC-V (REPLICATE_CPU_ROWS >= FACE_ROWS) and each core has one block of
# columns (no slot reuse); otherwise the reader builds the whole slot.
REPLICATE_SPLIT = 1
SEM_FACE1_BUILT = 0

CB_BCAST_TILES = 0  # broadcast tiles: row 0 fetched + replicated in place; also the write source

_cache = {}


def _div_up(a, b):
    return (a + b - 1) // b


def _split_core_count(grid, total_tiles):
    """Single source of the grid-size knob: cores offered to the work split."""
    target = grid.x * grid.y
    if MAX_CORES is not None:
        target = min(target, MAX_CORES)
    target = min(target, _div_up(total_tiles, MIN_TILES_PER_CORE))
    return max(1, target)


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
    split_cores = _split_core_count(grid, total_tiles)
    split_grid = ttnn.num_cores_to_corerangeset(split_cores, grid, True)
    (num_cores, all_cores, core_group_1, core_group_2, n_g1, n_g2) = ttnn.split_work_to_cores(
        split_grid, total_tiles, row_wise=True
    )
    range_sizes = []
    for group, n_t in ((core_group_1, n_g1), (core_group_2, n_g2)):
        if n_t == 0:
            continue
        range_sizes += [n_t] * len(_ordered_cores(group))
    cores = _ordered_cores(all_cores)
    assert len(cores) == len(range_sizes) == num_cores
    stride = CORE_ORDER_STRIDE
    while math.gcd(stride, num_cores) != 1:
        stride += 1
    assignments = []  # (core, start_t, n_t)
    start_t = 0
    for k, n_t in enumerate(range_sizes):
        assignments.append((cores[(k * stride) % num_cores], start_t, n_t))
        start_t += n_t
    assert start_t == total_tiles, (start_t, total_tiles)

    max_cols_per_core = max((s + n - 1) // Ht - s // Ht + 1 for _, s, n in assignments)
    col_block = min(max_cols_per_core, COL_BLOCK_CAP)
    num_col_blocks_max = _div_up(max_cols_per_core, col_block)
    bcast_depth = 1 if num_col_blocks_max == 1 else 2
    replicate_split = int(bool(REPLICATE_SPLIT) and REPLICATE_CPU_ROWS >= FACE_ROWS and num_col_blocks_max == 1)

    # ---- CB ----
    slot_bytes = page_size // 2 if HALF_TILE else page_size  # one CB page = one broadcast slot
    cb_bcast_tiles = ttnn.CBDescriptor(
        total_size=bcast_depth * col_block * slot_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_BCAST_TILES,
                data_format=input_tensor.dtype,
                page_size=slot_bytes,
            )
        ],
    )

    # ---- Kernels ----
    reader_ct_args = [
        CB_BCAST_TILES,
        Ht,
        col_block,
        page_size,
        row_fetch_bytes,
        face_bytes,
        face_row_bytes,
        REPLICATE_CPU_ROWS,
        READER_WRITE_EIGHTHS,
        Wt,
        int(bool(HALF_TILE)),
        ROW_FETCH_REQUESTS,
        int(bool(REPLICATE_NOC_FANOUT)),
        replicate_split,
        SEM_FACE1_BUILT,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args = [
        CB_BCAST_TILES,
        Ht,
        Wt,
        page_size,
        READER_WRITE_EIGHTHS,
        int(bool(HALF_TILE)),
        replicate_split,
        SEM_FACE1_BUILT,
        face_bytes,
        face_row_bytes,
        row_fetch_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

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
        common_runtime_args=[input_tensor.buffer_address(), output_tensor.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "g6_expand_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        common_runtime_args=[output_tensor.buffer_address(), input_tensor.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel],
        semaphores=[ttnn.SemaphoreDescriptor(id=SEM_FACE1_BUILT, core_ranges=all_cores, initial_value=0)],
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
        "replicate_cpu_rows": REPLICATE_CPU_ROWS,
        "core_order_stride": stride,
        "reader_write_eighths": READER_WRITE_EIGHTHS,
        "split_cores": split_cores,
        "max_cores": MAX_CORES,
        "min_tiles_per_core": MIN_TILES_PER_CORE,
        "half_tile": int(bool(HALF_TILE)),
        "slot_bytes": slot_bytes,
        "row_fetch_requests": ROW_FETCH_REQUESTS,
        "replicate_noc_fanout": int(bool(REPLICATE_NOC_FANOUT)),
        "replicate_split": replicate_split,
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
    kernels[0].common_runtime_args = [input_tensor.buffer_address(), output_tensor.buffer_address()]
    kernels[1].common_runtime_args = [output_tensor.buffer_address(), input_tensor.buffer_address()]
    descriptor.kernels = kernels
    return descriptor, plan
