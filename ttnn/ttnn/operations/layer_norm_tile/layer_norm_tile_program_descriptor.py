# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""layer_norm ProgramDescriptor: regime R1 (row_resident), tile-rows split across the compute grid."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# Blocking-model host constants (op_design.md, Blocking Model).
MIN_BLOCKS_PER_CORE = 2
IO_CHUNK_TARGET_TILES = 8
# Stream knobs (Refinement 1, measured on BH p100a). READ_TRID_DEPTH > 1 slowed L1-sourced rows and
# left DRAM rows flat; WRITE_CHUNKS_PER_FLUSH and BANK_STAGGER were flat (7 DRAM banks already
# spread the 24 * t row starts). All three are parked at their trivial settings.
READ_TRID_DEPTH = 1  # chunks kept in flight on the read trid ring, <= 15
WRITE_CHUNKS_PER_FLUSH = 1  # output chunks issued per flush + pop
BANK_STAGGER = False  # rotate each core's in-chunk tile order by its core index
# split_reader: the residual stream moves to the writer RISC, on the NoC opposite the reader's. Gated to
# at least half the grid active (it lost ~1.5 us on a 32-core grid). Any memory tiers: under the per-core
# NoC plan it measured flat on input-L1 / residual-DRAM and -7% on input-DRAM / residual-L1 (BH p100a).
RESIDUAL_ON_WRITER = True
OUT_DEPTH = 1
# COMPUTE_DEST_LANES is the tiles per DEST acquire on the streaming passes, clamped on device to DEST
# capacity (Refinement 2, BH p100a: 4 cut center 1.6k -> 0.9k and the affine passes 6.1k -> 5.0k
# cycles). A DEST-reuse fused normalize/affine pass measured slower under fp32 DEST in Refinement 2
# and again in Perf 1, and was removed.
COMPUTE_DEST_LANES = 4
L1_HEADROOM_BYTES = 64 * 1024

CB_INPUT_TILES = 0
CB_RESIDUAL_TILES = 1
CB_GAMMA_TILES = 2
CB_BETA_TILES = 3
CB_ROW_WORK_A = 4
CB_ROW_WORK_B = 5
CB_ROW_MEAN = 6
CB_INV_STD = 8
CB_AFFINE_SCRATCH = 9  # reader-private landing block of the gamma/beta multicast
CB_OUTPUT_TILES = 16

# Work-pair page format: one id per allocation group, the alias partner shares it (l1_ledger.md).
_WORK_ALIAS = {CB_ROW_WORK_A: CB_ROW_WORK_B}


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Blocking:
    block_row_tiles: int  # R
    io_chunk_tiles: int  # C
    in_depth: int
    tensor_w_tiles: int  # Wt
    has_residual: bool
    fp32_dest_acc_en: bool
    read_trid_depth: int
    write_chunks_per_flush: int
    bank_stagger: bool
    residual_on_writer: bool
    all_cores: object
    assignment: tuple  # ((core, row_tile_start, core_row_tiles), ...)


def tensor_row_tiles(shape):
    """Per-image ceil: each (..., H, W) image is tile-padded independently."""
    leading = math.prod(shape[:-2]) if len(shape) > 2 else 1
    return leading * _div_up(shape[-2], TILE)


def _affine_scratch_bytes(Wt, affine_is_tile):
    """gamma + beta compact multicast block: per tile column two 64 B tile rows (TILE) or one 64 B stick span."""
    entry_bytes = 2 * TILE * 2 if affine_is_tile else TILE * 2
    return 2 * Wt * entry_bytes


def _cb_pages(R, in_depth, *, Wt, has_residual):
    """Page count per CB: the single source for both the L1 footprint and the allocation (l1_ledger.md)."""
    pages = {
        CB_INPUT_TILES: in_depth * R * Wt,
        CB_GAMMA_TILES: Wt,
        CB_BETA_TILES: Wt,
        CB_ROW_WORK_A: R * Wt,  # alias group with CB_ROW_WORK_B: one allocation
        CB_ROW_MEAN: R,
        CB_INV_STD: R,
        CB_OUTPUT_TILES: OUT_DEPTH * R * Wt,
    }
    if has_residual:
        pages[CB_RESIDUAL_TILES] = in_depth * R * Wt
    return pages


def _cb_page_formats(fp32_dest_acc_en):
    io = (ttnn.tile_size(ttnn.bfloat16), ttnn.bfloat16)
    work_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    work = (ttnn.tile_size(work_dtype), work_dtype)
    return {
        CB_INPUT_TILES: io,
        CB_RESIDUAL_TILES: io,
        CB_GAMMA_TILES: io,
        CB_BETA_TILES: io,
        CB_ROW_WORK_A: work,
        CB_ROW_WORK_B: work,
        CB_ROW_MEAN: work,
        CB_INV_STD: work,
        CB_OUTPUT_TILES: io,
    }


def _footprint(pages, formats):
    return sum(num_pages * formats[index][0] for index, num_pages in pages.items())


def _split_rows(device, num_row_tiles):
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, group_1, group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, num_row_tiles, row_wise=True)
    assignment = []
    start = 0
    for group, per_core in ((group_1, rows_g1), (group_2, rows_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core
    return all_cores, tuple(assignment), rows_g1


def _io_chunk_tiles(Wt):
    return max(c for c in range(1, min(Wt, IO_CHUNK_TARGET_TILES) + 1) if Wt % c == 0)


def _residual_on_writer(residual, num_active_cores, num_grid_cores):
    if not RESIDUAL_ON_WRITER or residual is None:
        return False
    return 2 * num_active_cores >= num_grid_cores


def select_blocking(input_tensor, *, residual, fp32_dest_acc_en):
    """Single source of every block knob (R, C, in_depth, stream knobs). None when R1's predicate fails."""
    has_residual = residual is not None
    shape = list(input_tensor.shape)
    if shape[-1] % TILE != 0:
        return None
    Wt = shape[-1] // TILE
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    all_cores, assignment, core_row_tiles_max = _split_rows(device, tensor_row_tiles(shape))

    formats = _cb_page_formats(fp32_dest_acc_en)
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank - L1_HEADROOM_BYTES

    def in_depth_for(R):
        return 2 if _div_up(core_row_tiles_max, R) >= 2 else 1

    def fits(R):
        pages = _cb_pages(R, in_depth_for(R), Wt=Wt, has_residual=has_residual)
        return _footprint(pages, formats) + _affine_scratch_bytes(Wt, affine_is_tile=True) <= budget

    R = next((r for r in range(_div_up(core_row_tiles_max, MIN_BLOCKS_PER_CORE), 0, -1) if fits(r)), None)
    if R is None:
        return None

    return Blocking(
        block_row_tiles=R,
        io_chunk_tiles=_io_chunk_tiles(Wt),
        in_depth=in_depth_for(R),
        tensor_w_tiles=Wt,
        has_residual=has_residual,
        fp32_dest_acc_en=fp32_dest_acc_en,
        read_trid_depth=READ_TRID_DEPTH,
        write_chunks_per_flush=WRITE_CHUNKS_PER_FLUSH,
        bank_stagger=BANK_STAGGER,
        residual_on_writer=_residual_on_writer(residual, len(assignment), grid.x * grid.y),
        all_cores=all_cores,
        assignment=assignment,
    )


def _core_nocs(blocking, grid_y):
    """(reader_noc, writer_noc) per assignment entry; the kernels address through noc_index, so this is host-only.

    Under all-core load NoC0 reads have a y-gradient (BH p100a: first-row read ~25 us at the top row vs
    ~7.5 us at the bottom) and NoC1 carries every write. While most cores own one row, the heavy
    (larger-share) cores read on the otherwise idle NoC1 and the bottom half of the grid, whose reads
    finish first, writes on NoC0 (Perf 1: 8x512 focus 46.4 -> 36.9 us). Carve-out: when most cores own
    >= 2 rows, writes of row k overlap NoC0 reads of row k+1 and the swap measured +10%.
    """
    base = [(0, 1)] * len(blocking.assignment)
    rows = [core_row_tiles for _, _, core_row_tiles in blocking.assignment]
    if 2 * rows.count(1) <= len(rows):
        return base
    heavy = max(rows) if len(set(rows)) > 1 else None
    nocs = []
    for core, _, core_row_tiles in blocking.assignment:
        reader_noc = 1 if core_row_tiles == heavy else 0
        writer_noc = 0 if 2 * core.y >= grid_y else 1
        if blocking.residual_on_writer and reader_noc == writer_noc:
            # Both RISCs read here, and both use read cmd buf 1: they need different NoCs. Writer choice wins.
            reader_noc = 1 - writer_noc
        nocs.append((reader_noc, writer_noc))
    # The affine injector (last core) multicasts from its reader RISC; it keeps the default pair, the only
    # one its multicast was validated on (a swapped injector hung the writer's write barrier).
    nocs[-1] = (0, 1)
    return nocs


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _affine_scratch_cb(blocking, affine_is_tile):
    scratch_bytes = _affine_scratch_bytes(blocking.tensor_w_tiles, affine_is_tile)
    return ttnn.CBDescriptor(
        total_size=scratch_bytes,
        core_ranges=blocking.all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=CB_AFFINE_SCRATCH, data_format=ttnn.bfloat16, page_size=scratch_bytes)
        ],
    )


def _cb_descriptors(blocking):
    R = blocking.block_row_tiles
    Wt = blocking.tensor_w_tiles
    cores = blocking.all_cores
    pages = _cb_pages(R, blocking.in_depth, Wt=Wt, has_residual=blocking.has_residual)
    formats = _cb_page_formats(blocking.fp32_dest_acc_en)

    cbs = []
    for index, num_pages in pages.items():
        members = [index] + ([_WORK_ALIAS[index]] if index in _WORK_ALIAS else [])
        page_size = formats[index][0]
        cbs.append(
            ttnn.CBDescriptor(
                total_size=num_pages * page_size,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=member, data_format=formats[member][1], page_size=page_size)
                    for member in members
                ],
            )
        )
    return cbs


def create_program_descriptor(
    input_tensor, gamma, beta, residual, output_tensor, blocking, *, epsilon, compute_kernel_config
):
    width = list(input_tensor.shape)[-1]
    has_residual = blocking.has_residual
    R = blocking.block_row_tiles
    C = blocking.io_chunk_tiles
    Wt = blocking.tensor_w_tiles
    cores = blocking.all_cores

    input_tile_bytes = input_tensor.tile.get_tile_size(input_tensor.dtype)
    output_tile_bytes = output_tensor.tile.get_tile_size(output_tensor.dtype)
    affine_is_tile = gamma.layout == ttnn.TILE_LAYOUT
    # Elements per affine page: a tile row for TILE, the stick width for ROW_MAJOR.
    affine_page_elems = TILE if affine_is_tile else list(gamma.shape)[-1]

    reader_ct_args = [
        CB_INPUT_TILES,
        CB_RESIDUAL_TILES,
        CB_GAMMA_TILES,
        CB_BETA_TILES,
        R,
        Wt,
        C,
        int(has_residual),
        int(affine_is_tile),
        affine_page_elems,
        input_tile_bytes,
        gamma.buffer_aligned_page_size(),
        gamma.element_size(),
        blocking.read_trid_depth,
        int(blocking.residual_on_writer),
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(residual).get_compile_time_args() if has_residual else [0, 0])
    reader_ct_args.extend(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(beta).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        R,
        Wt,
        C,
        output_tile_bytes,
        blocking.write_chunks_per_flush,
        int(blocking.residual_on_writer),
        CB_RESIDUAL_TILES,
        blocking.in_depth,
        blocking.read_trid_depth,
        input_tile_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(residual).get_compile_time_args() if blocking.residual_on_writer else [0, 0]
    )

    compute_ct_args = [
        CB_INPUT_TILES,
        CB_RESIDUAL_TILES,
        CB_GAMMA_TILES,
        CB_BETA_TILES,
        CB_ROW_WORK_A,
        CB_ROW_WORK_B,
        CB_ROW_MEAN,
        CB_INV_STD,
        CB_OUTPUT_TILES,
        R,
        Wt,
        int(has_residual),
        width,
        _float_bits(1.0 / width),
        _float_bits(epsilon),
        COMPUTE_DEST_LANES,
    ]

    compute_rt = ttnn.RuntimeArgs()
    groups = {}  # (reader_noc, writer_noc) -> (cores, reader_rt, writer_rt)
    in_addr = input_tensor.buffer_address()
    res_addr = residual.buffer_address() if has_residual else 0
    gamma_addr = gamma.buffer_address()
    beta_addr = beta.buffer_address()
    out_addr = output_tensor.buffer_address()
    nocs = _core_nocs(blocking, input_tensor.device().compute_with_storage_grid_size().y)
    for core_idx, (core, row_tile_start, core_row_tiles) in enumerate(blocking.assignment):
        rotation = core_idx % C if blocking.bank_stagger else 0
        group_cores, reader_rt, writer_rt = groups.setdefault(
            nocs[core_idx], ([], ttnn.RuntimeArgs(), ttnn.RuntimeArgs())
        )
        group_cores.append(core)
        reader_rt[core.x][core.y] = [
            in_addr,
            res_addr,
            gamma_addr,
            beta_addr,
            row_tile_start,
            core_row_tiles,
            rotation,
        ]
        writer_rt[core.x][core.y] = [out_addr, row_tile_start, core_row_tiles, rotation, res_addr]
        compute_rt[core.x][core.y] = [core_row_tiles]

    noc_enum = (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)
    kernels = []
    readers = []
    for (reader_noc, writer_noc), (group_cores, reader_rt, writer_rt) in groups.items():
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in group_cores])
        readers.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "layer_norm_tile_reader.cpp"),
                core_ranges=ranges,
                compile_time_args=reader_ct_args,
                named_compile_time_args=[("affine_scratch_cb", CB_AFFINE_SCRATCH)],
                runtime_args=reader_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=noc_enum[reader_noc]
                ),
            )
        )
        kernels += [
            readers[-1],
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "layer_norm_tile_writer.cpp"),
                core_ranges=ranges,
                compile_time_args=writer_ct_args,
                runtime_args=writer_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=noc_enum[writer_noc]
                ),
            ),
        ]
    kernels.append(
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "layer_norm_tile_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        )
    )
    descriptor = ttnn.ProgramDescriptor(cbs=_cb_descriptors(blocking) + [_affine_scratch_cb(blocking, affine_is_tile)])
    # gamma/beta do not vary along the row split: one injector (the last core of the split) reads them
    # and multicasts the compact block to the active cores (Perf 1: affine TILE in DRAM 76.1 -> 52.9 us).
    # No handshake: the landing block is a fresh reader-private CB. Cores without rows are not in
    # `cores`, so they are neither receivers nor in the program. The send runs on the injector's
    # reader NoC, which sets the rectangle corner order.
    injector = blocking.assignment[-1][0]
    affine_mcast = ttnn.Mcast(
        input_tensor.device(),
        ttnn.McastConfig(handshake=False, noc=noc_enum[nocs[-1][0]]),
        cores,
        cores.num_cores(),
        ttnn.McastExplicitFixedSenderConfig([injector]),
    )
    affine_mcast.attach(descriptor, "amc", readers, 0)
    descriptor.kernels = kernels
    return descriptor
