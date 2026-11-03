# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""linear_gelu ProgramDescriptor: regime R1 (weight-stationary 2D grid).

m (token tile rows) over logical rows, n (output tiles) over logical columns; logical rows are the grid rows, or the
grid columns when that fills more cores (short m). One core per logical column (staggered) reads the column's W
slice (plus the bias as the 25th K tile row) chunk by chunk and multicasts it along the column into
cb_weight_resident, where it stays for every row-block. x row-blocks are multicast along each logical row. One
K-block of k_ext_tiles accumulates in fp32 DEST; GELU runs on the packer before each subblock is packed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
HIDDEN = 768
FFN = 3072
K_TILES = HIDDEN // TILE
N_TILES = FFN // TILE
# x/W K tiles plus the bias K tile (x side: the ones tile).
K_EXT_TILES = K_TILES + 1

# W chunk width (n tiles per mcast round). 1 measured fastest at every shape: the first chunk (25 tiles) lands
# 3x sooner than at 3, which is the fill latency in front of all compute, and the finest n split also fills the grid
# best. The 4-tile fp32 DEST is still filled by a 4x1 subblock.
BLOCK_W = 1
# W chunk j of logical column g is read and multicast by row (j + g + W_ROUND_SHIFT) mod active_rows (rotating
# sender offset by the column), so no grid line's NoC links carry every column's chunk. The shift of 1 keeps a
# column's chunk-0 owner off its row's x round-0 sender (x round offset = row), so the two first reads overlap.
W_SENDER_STAGGER = True
W_ROUND_SHIFT = 1
# W read lookahead in rounds: chunks < W_PREFETCH_ROUNDS are read at kernel start, chunk j + W_PREFETCH_ROUNDS at the
# start of round j, so each read lands during the previous round's multicast. 1 measured best: 8x512 94.6 -> 92.4,
# 2x512 36.3 -> 33.1 us; 2 / 3 rounds 94.8 / 96.3 us (reads in flight contend with the column multicast).
W_PREFETCH_ROUNDS = 1
# W chunks multicast before x row-block 0, by x's buffer type; the x round-0 sender's reads are issued first and land
# meanwhile. Either order serializes the two first multicasts (each channel's handshake waits for the other), so this
# only pays where x's reads are slow: DRAM x 8x512 143.4 -> 141.0 us at 1; L1 x flat at 8x512, +0.6 us at 2x512.
W_CHUNKS_BEFORE_X = {ttnn.BufferType.L1: 0, ttnn.BufferType.DRAM: 1}
ACT_DEPTH = 2
# Upper bound on block_h (x row-block rows); the L1 solve may still pick fewer. With the load-macro GELU the matmul
# co-binds, and a taller block hides more of the W-paced row-block 0 behind compute: 3 -> 4 is 8x512 98.5 -> 95.1,
# 8x384 79.9 -> 76.1 us (medians of 3).
BLOCK_H_MAX = 4
# Carve-out: DRAM output is writer-bound and block_h 4 measured 143.8 -> 153.7 us there (8x512, medians of 3).
BLOCK_H_MAX_DRAM_OUT = 3
# Minimum cb_output_tiles ring in pages; the depth in (block_h x block_w) n-blocks derives from it, so the ring stays a
# whole number of blocks. The writer needs about 24 pages to ride out DRAM write latency: 8 measured 143..146 us at
# 8x512 DRAM out vs 138.7, and block_h 3 at 18 pages 138.6 vs 137.5 at 24.
OUT_RING_PAGES = 24
CB_BUDGET_BYTES = 900 * 1024

CB_ACTIVATION_BLOCK = 0
CB_WEIGHT_RESIDENT = 1
CB_OUTPUT_TILES = 2
CB_BIAS_ONES_TILE = 3

W_MCAST_PREFIX = "wmc"
X_MCAST_PREFIX = "xmc"


@dataclass(frozen=True)
class Blocking:
    tensor_m_tiles: int
    core_m_tiles: int
    active_rows: int
    core_n_blocks: int
    core_n_tiles_max: int
    active_cols: int
    block_h: int
    dest_tiles: int
    block_w: int
    # Logical rows (m) on physical grid columns (x) instead of grid rows (y): fills more cores when m is short.
    transposed: bool


def _div_up(a, b):
    return -(-a // b)


def _out_depth(block_h, block_w):
    return max(2, _div_up(OUT_RING_PAGES, block_h * block_w))


# Row-block split by x's buffer type. "balanced": one block_h block (its compute hides the W delivery), then the rest
# split evenly, so no 1-row tail runs 1x1 subblocks: DRAM x 8x512 99.1 -> 95.5, 8x384 79.8 -> 77.0 us. Carve-out
# "tail" (block_h blocks plus one remainder block) for L1 x, where balanced measured 8x512 89.8 -> 90.6 us: x lands
# early there, and the final 3-tile subblock's GELU drains unhidden.
ROW_SPLIT = {ttnn.BufferType.L1: "tail", ttnn.BufferType.DRAM: "balanced"}


def _num_full(rows, block_h, policy):
    """Leading block_h-row blocks before the balanced rest."""
    num_blocks = _div_up(rows, block_h)
    return num_blocks - 1 if policy == "tail" else min(1, num_blocks - 1)


def row_blocks(rows, block_h, num_full):
    """(num_blocks, per-block rows, per-block kind) mirroring kernels/linear_gelu_row_blocks.h: num_full block_h-row
    blocks, then the rest split evenly, larger first. kind 0 = block_h rows, 1 = q + 1 rows, 2 = q rows."""
    num_blocks = _div_up(rows, block_h)
    rest_blocks = num_blocks - num_full
    q, r = divmod(rows - num_full * block_h, rest_blocks)
    sizes = [block_h] * num_full + [q + (1 if j < r else 0) for j in range(rest_blocks)]
    kinds = [0] * num_full + [1 if j < r else 2 for j in range(rest_blocks)]
    return num_blocks, q, sizes, kinds


def _footprint_bytes(block_h, core_n_tiles_max, block_w, tile_bytes):
    fixed = (K_EXT_TILES * core_n_tiles_max + 1) * tile_bytes
    act = ACT_DEPTH * block_h * K_EXT_TILES * tile_bytes
    out = _out_depth(block_h, block_w) * block_h * block_w * tile_bytes
    return fixed + act + out


def _candidate(tensor_m_tiles, block_w, m_slots, n_slots, tile_bytes, block_h_max):
    """Split m over m_slots grid lines and N_TILES / block_w n-blocks over n_slots; None if the W slice won't fit."""
    n_blocks = N_TILES // block_w
    core_n_blocks = _div_up(n_blocks, n_slots)
    core_n_tiles_max = block_w * core_n_blocks
    core_m_tiles = _div_up(tensor_m_tiles, m_slots)
    fits = [
        h
        for h in range(1, min(core_m_tiles, block_h_max) + 1)
        if _footprint_bytes(h, core_n_tiles_max, block_w, tile_bytes) <= CB_BUDGET_BYTES
    ]
    if not fits:
        return None
    return dict(
        core_m_tiles=core_m_tiles,
        active_rows=_div_up(tensor_m_tiles, core_m_tiles),
        core_n_blocks=core_n_blocks,
        core_n_tiles_max=core_n_tiles_max,
        active_cols=_div_up(n_blocks, core_n_blocks),
        block_h=max(fits),
        block_w=block_w,
    )


def select_blocking(input_tensor, compute_kernel_config, out_buffer_type):
    grid = input_tensor.device().compute_with_storage_grid_size()
    batch, _, seq, _ = list(input_tensor.shape)
    tensor_m_tiles = batch * _div_up(seq, TILE)
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    assert N_TILES % BLOCK_W == 0
    block_h_max = BLOCK_H_MAX_DRAM_OUT if out_buffer_type == ttnn.BufferType.DRAM else BLOCK_H_MAX
    # Orientation by output tiles per core (the per-tile GELU binds); ties keep m on grid rows.
    best = None
    for transposed in (False, True):
        m_slots, n_slots = (grid.x, grid.y) if transposed else (grid.y, grid.x)
        cand = _candidate(tensor_m_tiles, BLOCK_W, m_slots, n_slots, tile_bytes, block_h_max)
        if cand is None:
            continue
        tiles = cand["core_m_tiles"] * cand["core_n_tiles_max"]
        if best is None or tiles < best[0]:
            best = (tiles, transposed, cand)
    if best is None:
        raise RuntimeError(f"linear_gelu: no W slice split fits {CB_BUDGET_BYTES} B of CBs on a {grid.x}x{grid.y} grid")
    _, transposed, cand = best
    # fp32 DEST halves the half-sync DEST capacity.
    dest_tiles = 4 if compute_kernel_config.fp32_dest_acc_en else 8
    return Blocking(tensor_m_tiles=tensor_m_tiles, dest_tiles=dest_tiles, transposed=transposed, **cand)


def _subblock(rows, block_w, dest_tiles):
    """Largest-area (sb_h, sb_w) with sb_h | rows, sb_w | block_w, area <= dest_tiles; larger sb_h on ties."""
    best = (1, 1)
    for sb_h in range(1, rows + 1):
        if rows % sb_h:
            continue
        for sb_w in range(1, block_w + 1):
            if block_w % sb_w or sb_h * sb_w > dest_tiles:
                continue
            if (sb_h * sb_w, sb_h) > (best[0] * best[1], best[0]):
                best = (sb_h, sb_w)
    return best


def _cb(index, num_pages, tile_bytes, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * tile_bytes,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=tile_bytes)
        ],
    )


def create_program_descriptor(input_tensor, weight, bias, output, compute_kernel_config):
    device = input_tensor.device()
    blocking = select_blocking(input_tensor, compute_kernel_config, output.memory_config().buffer_type)
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    block_h = blocking.block_h
    block_w = blocking.block_w
    transposed = blocking.transposed

    def phys(row, col):
        return (row, col) if transposed else (col, row)

    last_x, last_y = phys(blocking.active_rows - 1, blocking.active_cols - 1)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(last_x, last_y))])
    # A logical column (W group) is a physical column, or a physical row when transposed; x groups are the other.
    col_order = ttnn.McastCoreOrder.RowMajor if transposed else ttnn.McastCoreOrder.ColumnMajor
    row_order = ttnn.McastCoreOrder.ColumnMajor if transposed else ttnn.McastCoreOrder.RowMajor

    reader_ct = [
        CB_ACTIVATION_BLOCK,
        CB_WEIGHT_RESIDENT,
        CB_BIAS_ONES_TILE,
        block_h,
        block_w,
        K_TILES,
        K_EXT_TILES,
        N_TILES,
        tile_bytes,
        W_PREFETCH_ROUNDS,
        W_CHUNKS_BEFORE_X[input_tensor.memory_config().buffer_type],
    ]
    for tensor in (input_tensor, weight, bias):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    seq = list(input_tensor.shape)[-2]
    writer_ct = [CB_OUTPUT_TILES, block_h, block_w, N_TILES, tile_bytes, _div_up(seq, TILE), seq % TILE]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    compute_ct = [CB_ACTIVATION_BLOCK, CB_WEIGHT_RESIDENT, CB_OUTPUT_TILES, block_h, block_w, K_EXT_TILES]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    core_n_tiles = blocking.core_n_tiles_max
    per_row = []
    areas = set()
    row_split = ROW_SPLIT[input_tensor.memory_config().buffer_type]
    for row in range(blocking.active_rows):
        rows = min(blocking.core_m_tiles, blocking.tensor_m_tiles - row * blocking.core_m_tiles)
        num_full = _num_full(rows, block_h, row_split)
        num_blocks, q, sizes, kinds = row_blocks(rows, block_h, num_full)
        sbs = [_subblock(h, block_w, blocking.dest_tiles) if h else (1, 1) for h in (block_h, q + 1, q)]
        for kind in kinds:
            areas.add(sbs[kind][0] * sbs[kind][1])
        per_row.append((rows, num_blocks, num_full, [v for sb in sbs for v in sb]))
    # Every block starts on a multiple of its subblock area (kernel pad), so the ring must be a multiple of all.
    out_pages = _out_depth(block_h, block_w) * block_h * block_w
    lcm = math.lcm(*areas)
    out_pages = _div_up(out_pages, lcm) * lcm
    compute_ct.append(out_pages)
    writer_ct.insert(7, out_pages)
    for row in range(blocking.active_rows):
        row_start = row * blocking.core_m_tiles
        rows, num_blocks, num_full, sbs = per_row[row]
        for col in range(blocking.active_cols):
            n_tile_start = col * core_n_tiles
            num_n_blocks = min(core_n_tiles, N_TILES - n_tile_start) // block_w
            x, y = phys(row, col)
            reader_rt[x][y] = [
                input_tensor.buffer_address(),
                weight.buffer_address(),
                bias.buffer_address(),
                row_start,
                rows,
                n_tile_start,
                num_n_blocks,
                row,
                col + W_ROUND_SHIFT if W_SENDER_STAGGER else 0,
                num_blocks,
                num_full,
            ]
            writer_rt[x][y] = [
                output.buffer_address(),
                row_start,
                rows,
                n_tile_start,
                num_n_blocks,
                num_blocks,
                num_full,
                *sbs,
            ]
            compute_rt[x][y] = [rows, num_n_blocks, num_blocks, num_full, *sbs]
    cbs = [
        _cb(CB_ACTIVATION_BLOCK, ACT_DEPTH * block_h * K_EXT_TILES, tile_bytes, cores),
        _cb(CB_WEIGHT_RESIDENT, K_EXT_TILES * blocking.core_n_tiles_max, tile_bytes, cores),
        _cb(CB_OUTPUT_TILES, out_pages, tile_bytes, cores),
        _cb(CB_BIAS_ONES_TILE, 1, tile_bytes, cores),
    ]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "linear_gelu_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "linear_gelu_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "linear_gelu_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    # One group per logical column (group = its active rows); the sender rotates with the W chunk.
    w_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
        cores,
        blocking.active_rows,
        # A one-row column has no other sender to rotate to (should_send is then always true on it).
        ttnn.McastRotatingSenderConfig() if blocking.active_rows > 1 else ttnn.McastFixedSenderConfig(sender_index=0),
        col_order,
    )
    w_mcast.attach(descriptor, W_MCAST_PREFIX, [reader], 0)
    # R2: x row-blocks are shared by a grid row; one group per row, the sender rotates with the row-block.
    x_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
        cores,
        blocking.active_cols,
        ttnn.McastRotatingSenderConfig(),
        row_order,
    )
    x_mcast.attach(descriptor, X_MCAST_PREFIX, [reader], w_mcast.next_semaphore_id())
    descriptor.kernels = [reader, writer, compute]
    return descriptor
