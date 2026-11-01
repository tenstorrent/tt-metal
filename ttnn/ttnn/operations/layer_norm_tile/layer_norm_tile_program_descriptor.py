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
READ_TRID_DEPTH = 1  # chunks kept in flight on the read trid ring, <= 14 (the affine scatter uses id depth + 1)
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
# Row statistics (Refinement 3, BH p100a). STREAM_ROW_STATS accumulates the row sums per stats window as the
# reader's chunks land and, with ONE_PASS_VARIANCE, the sum of squares too (var = E[d^2] - E[d]^2, no centered
# sum-of-squares pass). Streaming is gated to fp32 DEST + HiFi4 (one-pass precision: z * z exact, fp32
# accumulation; the streamed mean alone never measured a win) and to the split classes where it wins
# (_stream_row_stats); elsewhere the row is waited whole and the variance stays two-pass.
STREAM_ROW_STATS = True
ONE_PASS_VARIANCE = True
# Leading blocks per core whose stats stream (None: all). Later blocks mostly land under the previous block's
# compute; streaming them costs ~1 us per resident row in window round trips, yet 1 measured slower on
# focus1 (36.3 vs 35.9 us) because the first row's tail still waits on reads.
STREAMED_BLOCKS = None
# STREAM_NORMALIZE (one-pass only): center, normalize, scale and shift run per stats window, so the writer
# starts at the row's first window instead of during the last whole-row pass. Balanced splits only unless
# STREAM_NORMALIZE_UNBALANCED, which gives it to the heaviest cores (measured slower: focus1 37.1, focus2 ~51 us).
STREAM_NORMALIZE = True
STREAM_NORMALIZE_UNBALANCED = False
# Work unit (op_design.md R1 / R3, Refinement 4, BH p100a). BAND_ROWS = 16 splits work in 16-row half-tile
# bands where _half_band_split selects it (measured); 32 is R1 (whole tile-rows) everywhere.
BAND_ROWS = 16
# Run order of the band-count groups. None: measured gate (_even_count_first); True / False force it.
HALF_BAND_EVEN_FIRST = None
# Parked R3 knobs (measured on the 8x512 rows, 3-5 run medians):
#   HALF_BAND_NOC_PLAN = False (default NoCs on every core): focus1 50.8, focus2 52.7 us vs 35.4 / 47.3.
#   HEAVY_READER_NOC_BY_Y (heavy cores in the bottom half read on NoC0): focus2 46.9 vs 47.3, focus3 41.0 (flat).
#   HALF_BAND_STREAM (streamed stats + per-window normalize on the 2/3-band split): focus1 38.2-39.9,
#   focus2 48.4-51.8 us, worse on every row.
HALF_BAND_NOC_PLAN = True
HEAVY_READER_NOC_BY_Y = False
HALF_BAND_STREAM = False
L1_HEADROOM_BYTES = 64 * 1024

CB_INPUT_TILES = 0
CB_RESIDUAL_TILES = 1
CB_GAMMA_TILES = 2
CB_BETA_TILES = 3
CB_ROW_WORK_A = 4
CB_ROW_WORK_B = 5
CB_ROW_MEAN = 6
CB_STAT_PARTIAL = 7  # raw row sums carried between stats windows
CB_INV_STD = 8
CB_AFFINE_SCRATCH = 9  # reader-private landing block of the gamma/beta multicast
CB_CHUNK_A = 10  # stream_normalize: one window between per-window passes
CB_CHUNK_B = 11
CB_ROW_SHIFT = 12  # one_pass + residual: the row's first x tile, column 0 = each row's shift
CB_OUTPUT_TILES = 16
NUM_CIRCULAR_BUFFERS = 64

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
    stats_window_tiles: int  # G: tiles per row-stats window, a multiple of C or Wt
    one_pass: bool  # one-pass variance (sum and sum of squares in the stats windows)
    stream_normalize: bool  # per-window normalize + affine compiled in (one_pass only)
    stream_normalize_min_rows: int  # cores owning at least this many tile-rows take it
    half_band: bool  # R3: the work unit is a 16-row band, two per compute tile-row
    image_row_tiles: int  # tile-rows per (..., H, W) image
    image_bands: int  # valid (not all-padding) bands per image
    all_cores: object
    assignment: tuple  # ((core, unit_start, unit_count, core_row_tiles), ...)


def _num_images(shape):
    return math.prod(shape[:-2]) if len(shape) > 2 else 1


def tensor_row_tiles(shape):
    """Per-image ceil: each (..., H, W) image is tile-padded independently."""
    return _num_images(shape) * _div_up(shape[-2], TILE)


def _affine_scratch_bytes(Wt, affine_is_tile):
    """gamma + beta compact multicast block: per tile column two 64 B tile rows (TILE) or one 64 B stick span."""
    entry_bytes = 2 * TILE * 2 if affine_is_tile else TILE * 2
    return 2 * Wt * entry_bytes


def _cb_pages(R, in_depth, *, Wt, has_residual, stats_window_tiles, one_pass, stream_normalize, whole_row_passes):
    """Page count per CB: the single source for both the L1 footprint and the allocation (l1_ledger.md)."""
    pages = {
        CB_INPUT_TILES: in_depth * R * Wt,
        CB_GAMMA_TILES: Wt,
        CB_BETA_TILES: Wt,
        CB_ROW_MEAN: R,
        CB_INV_STD: R,
        CB_OUTPUT_TILES: OUT_DEPTH * R * Wt,
    }
    if has_residual:
        pages[CB_RESIDUAL_TILES] = in_depth * R * Wt
    if has_residual or whole_row_passes:
        pages[CB_ROW_WORK_A] = R * Wt  # z, or the whole-row passes; alias group with CB_ROW_WORK_B
    if stream_normalize:
        pages[CB_CHUNK_A] = stats_window_tiles
        pages[CB_CHUNK_B] = stats_window_tiles
    if stats_window_tiles < Wt:
        pages[CB_STAT_PARTIAL] = 2 if one_pass else 1
    if one_pass and has_residual:
        pages[CB_ROW_SHIFT] = 1
    return pages


def _cb_page_formats(fp32_dest_acc_en):
    io = (ttnn.tile_size(ttnn.bfloat16), ttnn.bfloat16)
    work_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    work = (ttnn.tile_size(work_dtype), work_dtype)
    return {
        CB_INPUT_TILES: io,
        CB_ROW_SHIFT: io,
        CB_RESIDUAL_TILES: io,
        CB_GAMMA_TILES: io,
        CB_BETA_TILES: io,
        CB_ROW_WORK_A: work,
        CB_ROW_WORK_B: work,
        CB_ROW_MEAN: work,
        CB_STAT_PARTIAL: work,
        CB_CHUNK_A: work,
        CB_CHUNK_B: work,
        CB_INV_STD: work,
        CB_OUTPUT_TILES: io,
    }


def _footprint(pages, formats):
    return sum(num_pages * formats[index][0] for index, num_pages in pages.items())


def image_bands(shape):
    """Valid 16-row bands per image: the last tile-row's bottom band is all padding when H % 32 in [1, 16]."""
    return 2 * _div_up(shape[-2], TILE) - (1 if 0 < shape[-2] % TILE <= TILE // 2 else 0)


def _split_units(device, num_units, half_band, even_first):
    """Contiguous unit runs per core: ((core, unit_start, unit_count, compute_rows), ...).

    Half-band: an odd-count run holds one lone band (RowSources). even_first runs the even-count group first,
    so every even-count run starts on a tile-row and moves whole tiles."""
    grid = device.compute_with_storage_grid_size()
    (_, all_cores, group_1, group_2, units_g1, units_g2) = ttnn.split_work_to_cores(grid, num_units, row_wise=True)
    groups = [(group_1, units_g1), (group_2, units_g2)]
    if half_band and even_first:
        groups.sort(key=lambda group: group[1] % 2)
    assignment = []
    start = 0
    for group, per_core in groups:
        if per_core == 0:
            continue
        rows = _div_up(per_core, 2) if half_band else per_core
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core, rows))
            start += per_core
    return all_cores, tuple(assignment)


def _half_band_split(shape, num_grid_cores):
    """R3 where it measured a win over R1 (8x512 rows ~1 us, under-filled rows 1.2-3.3 us):
    - more tile-rows than cores, not a multiple: R1 leaves a tail of cores with a whole extra tile-row;
      R3 cuts it to half a tile-row of bytes (128 rows on 110 cores: 2 rows -> 3 bands);
    - every band fits on its own core: the same balance on up to twice the cores.
    It lost where it turns a balanced R1 split into an unbalanced one (8x384 36.1 -> 38.4, 64x17 21.7 -> 22.7 us):
    the balanced split keeps the streamed row-stats path."""
    if BAND_ROWS != TILE // 2:
        return False
    num_row_tiles = tensor_row_tiles(shape)
    if num_row_tiles >= num_grid_cores:
        return num_row_tiles % num_grid_cores != 0
    return _num_images(shape) * image_bands(shape) <= num_grid_cores


def _even_count_first(residual):
    """Even-band-count cores first puts the 3-band cores at the grid bottom. Measured: residual in DRAM
    focus2 49.2 -> 47.3, affine TILE 50.9 -> 48.8, focus1 bf16 DEST 36.5 -> 35.3 us; without it (focus3
    33.2 -> 40.0, L1 no-residual 20.7 -> 21.1) or with the residual in L1 (24.5 -> 24.9) heavy-first wins."""
    if HALF_BAND_EVEN_FIRST is not None:
        return HALF_BAND_EVEN_FIRST
    return residual is not None and residual.memory_config().buffer_type == ttnn.BufferType.DRAM


def _io_chunk_tiles(Wt):
    return max(c for c in range(1, min(Wt, IO_CHUNK_TARGET_TILES) + 1) if Wt % c == 0)


def _is_balanced(assignment, half_band=False):
    if half_band and HALF_BAND_STREAM:
        return True
    return len({units for _, _, units, _ in assignment}) == 1


def _stream_row_stats(input_tensor, assignment, *, residual, compute_kernel_config, half_band):
    """Streamed one-pass stats where they win. On an unbalanced split the heavy cores set the end:
    - DRAM input: light cores finishing sooner write into the heavy cores' next-row DRAM reads
      (focus2 median 47.6 -> 48.9 us);
    - residual: the shifted residual add (an extra DEST-reuse pass per tile) costs more on the heavy cores'
      resident second row than streaming saves on the first (focus1 36.5 -> 36.7..37.9 us).
    Balanced splits stream (8x384 38.6 -> 36.1, 3x500 24.9 -> 22.2 us), except with the residual in L1: its
    reads finish early, so the shifted add's extra pass is not paid back (2x512 all L1 14.7 -> 14.9 us)."""
    if not STREAM_ROW_STATS or not _one_pass_capable(compute_kernel_config):
        return False
    dram_input = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    if residual is not None:
        return _is_balanced(assignment, half_band) and residual.memory_config().buffer_type == ttnn.BufferType.DRAM
    return _is_balanced(assignment, half_band) or not dram_input


def _stats_window_tiles(Wt, C, streamed):
    """Smallest even multiple of the reader chunk C dividing Wt (windows pair their tiles); Wt when odd or off."""
    if not streamed:
        return Wt
    if C % 2 == 0:
        return C
    return 2 * C if Wt % 2 == 0 else Wt


def _stream_normalize_min_rows(assignment, half_band):
    """Fewest tile-rows of a core that streams normalize: every core on a balanced split; on an unbalanced one
    none (or the heaviest under STREAM_NORMALIZE_UNBALANCED). With every core streaming, the light cores'
    earlier writes land on the heavy cores' next-row reads (focus1 36.2 -> 41 us)."""
    heaviest = max(rows for *_, rows in assignment)
    if half_band and HALF_BAND_STREAM:
        return 1
    return heaviest if _is_balanced(assignment) or STREAM_NORMALIZE_UNBALANCED else heaviest + 1


def _one_pass_capable(compute_kernel_config):
    return (
        bool(compute_kernel_config.fp32_dest_acc_en) and compute_kernel_config.math_fidelity == ttnn.MathFidelity.HiFi4
    )


def _residual_on_writer(residual, num_active_cores, num_grid_cores):
    if not RESIDUAL_ON_WRITER or residual is None:
        return False
    return 2 * num_active_cores >= num_grid_cores


def select_blocking(input_tensor, *, residual, compute_kernel_config):
    """Single source of every block knob (R, C, G, in_depth, stream knobs). None when R1's predicate fails."""
    has_residual = residual is not None
    fp32_dest_acc_en = bool(compute_kernel_config.fp32_dest_acc_en)
    shape = list(input_tensor.shape)
    if shape[-1] % TILE != 0:
        return None
    Wt = shape[-1] // TILE
    C = _io_chunk_tiles(Wt)
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    half_band = _half_band_split(shape, grid.x * grid.y)
    num_units = _num_images(shape) * image_bands(shape) if half_band else tensor_row_tiles(shape)
    all_cores, assignment = _split_units(device, num_units, half_band, _even_count_first(residual))
    core_row_tiles_max = max(rows for *_, rows in assignment)
    streamed = _stream_row_stats(
        input_tensor, assignment, residual=residual, compute_kernel_config=compute_kernel_config, half_band=half_band
    )
    G = _stats_window_tiles(Wt, C, streamed)
    one_pass = ONE_PASS_VARIANCE and streamed
    stream_min_rows = _stream_normalize_min_rows(assignment, half_band)
    stream_normalize = STREAM_NORMALIZE and one_pass and any(rows >= stream_min_rows for *_, rows in assignment)
    whole_row_passes = not stream_normalize or any(rows < stream_min_rows for *_, rows in assignment)

    formats = _cb_page_formats(fp32_dest_acc_en)
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank - L1_HEADROOM_BYTES

    def in_depth_for(R):
        return 2 if _div_up(core_row_tiles_max, R) >= 2 else 1

    def fits(R):
        pages = _cb_pages(
            R,
            in_depth_for(R),
            Wt=Wt,
            has_residual=has_residual,
            stats_window_tiles=G,
            one_pass=one_pass,
            stream_normalize=stream_normalize,
            whole_row_passes=whole_row_passes,
        )
        return _footprint(pages, formats) + _affine_scratch_bytes(Wt, affine_is_tile=True) <= budget

    R = next((r for r in range(_div_up(core_row_tiles_max, MIN_BLOCKS_PER_CORE), 0, -1) if fits(r)), None)
    if R is None:
        return None

    return Blocking(
        block_row_tiles=R,
        io_chunk_tiles=C,
        in_depth=in_depth_for(R),
        tensor_w_tiles=Wt,
        has_residual=has_residual,
        fp32_dest_acc_en=fp32_dest_acc_en,
        read_trid_depth=READ_TRID_DEPTH,
        write_chunks_per_flush=WRITE_CHUNKS_PER_FLUSH,
        bank_stagger=BANK_STAGGER,
        residual_on_writer=_residual_on_writer(residual, len(assignment), grid.x * grid.y),
        stats_window_tiles=G,
        one_pass=one_pass,
        stream_normalize=stream_normalize,
        stream_normalize_min_rows=stream_min_rows,
        half_band=half_band,
        image_row_tiles=_div_up(shape[-2], TILE),
        image_bands=image_bands(shape),
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
    rows = [core_row_tiles for *_, core_row_tiles in blocking.assignment]
    if 2 * rows.count(1) <= len(rows) or (blocking.half_band and not HALF_BAND_NOC_PLAN):
        return base
    heavy = max(rows) if len(set(rows)) > 1 else None
    nocs = []
    for core, *_, core_row_tiles in blocking.assignment:
        reader_noc = 1 if core_row_tiles == heavy else 0
        if HEAVY_READER_NOC_BY_Y and core_row_tiles == heavy and 2 * core.y >= grid_y:
            reader_noc = 0
        writer_noc = 0 if 2 * core.y >= grid_y else 1
        if blocking.residual_on_writer and reader_noc == writer_noc:
            # Both RISCs read here, and both use read cmd buf 1: they need different NoCs. Writer choice wins.
            reader_noc = 1 - writer_noc
        nocs.append((reader_noc, writer_noc))
    # The affine injector (last core) multicasts from its reader RISC; it keeps the default pair, the only
    # one its multicast was validated on (a swapped injector hung the writer's write barrier).
    nocs[-1] = (0, 1)
    return nocs


def _streamed_rows(blocking):
    """Leading tile-rows per core whose stats stream (STREAMED_BLOCKS whole blocks, or every row)."""
    if STREAMED_BLOCKS is None:
        return max(rows for *_, rows in blocking.assignment)
    return STREAMED_BLOCKS * blocking.block_row_tiles


def _compute_config(requested, blocking):
    """The caller's config plus UnpackToDestFp32 on the fp32 row-sum carry CB (lossless reload, not TF32)."""
    config = ttnn.ComputeConfigDescriptor()
    for field in (
        "math_fidelity",
        "fp32_dest_acc_en",
        "math_approx_mode",
        "dst_full_sync_en",
        "bfp8_pack_precise",
        "enable_trisc2_rvv",
        "unpack_to_dest_mode",
    ):
        setattr(config, field, getattr(requested, field))
    if blocking.fp32_dest_acc_en and blocking.stats_window_tiles < blocking.tensor_w_tiles:
        modes = list(config.unpack_to_dest_mode) or [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
        modes[CB_STAT_PARTIAL] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    return config


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
    pages = _cb_pages(
        R,
        blocking.in_depth,
        Wt=Wt,
        has_residual=blocking.has_residual,
        stats_window_tiles=blocking.stats_window_tiles,
        one_pass=blocking.one_pass,
        stream_normalize=blocking.stream_normalize,
        whole_row_passes=not blocking.stream_normalize
        or any(rows < blocking.stream_normalize_min_rows for *_, rows in blocking.assignment),
    )
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

    band_ct_args = [int(blocking.half_band), blocking.image_row_tiles, blocking.image_bands]
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
        *band_ct_args,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(residual).get_compile_time_args()
        if has_residual
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
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
        *band_ct_args,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(residual).get_compile_time_args()
        if blocking.residual_on_writer
        else ttnn.TensorAccessorArgs().get_compile_time_args()
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
        _float_bits(1.0 / width),
        _float_bits(epsilon),
        COMPUTE_DEST_LANES,
        blocking.stats_window_tiles,
        int(blocking.one_pass),
        CB_STAT_PARTIAL,
        int(blocking.stream_normalize),
        CB_CHUNK_A,
        CB_CHUNK_B,
        _streamed_rows(blocking),
        CB_ROW_SHIFT,
    ]

    compute_rt = ttnn.RuntimeArgs()
    groups = {}  # (reader_noc, writer_noc) -> (cores, reader_rt, writer_rt)
    in_addr = input_tensor.buffer_address()
    res_addr = residual.buffer_address() if has_residual else 0
    gamma_addr = gamma.buffer_address()
    beta_addr = beta.buffer_address()
    out_addr = output_tensor.buffer_address()
    nocs = _core_nocs(blocking, input_tensor.device().compute_with_storage_grid_size().y)
    for core_idx, (core, unit_start, unit_count, core_row_tiles) in enumerate(blocking.assignment):
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
            unit_start,
            unit_count,
            rotation,
        ]
        writer_rt[core.x][core.y] = [out_addr, unit_start, unit_count, rotation, res_addr]
        compute_rt[core.x][core.y] = [core_row_tiles, int(core_row_tiles >= blocking.stream_normalize_min_rows)]

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
            config=_compute_config(compute_kernel_config, blocking),
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
