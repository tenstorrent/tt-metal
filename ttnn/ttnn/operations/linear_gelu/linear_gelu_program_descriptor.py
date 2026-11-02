# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""linear_gelu ProgramDescriptor: regime R1 (weight-stationary 2D grid).

m (token tile rows) over grid rows, n (output tiles) over grid columns. Each column's row-0 core reads the
column's W slice (plus the bias as the 25th K tile row) chunk by chunk and multicasts it down the column into
cb_weight_resident, where it stays for every row-block. x row-blocks are read per core. One K-block of
k_ext_tiles accumulates in fp32 DEST; GELU runs on the packer before each subblock is packed.
"""

from __future__ import annotations

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

BLOCK_W = 3
ACT_DEPTH = 2
OUT_DEPTH = 2
CB_BUDGET_BYTES = 900 * 1024
# 0 = accurate tanh GELU (lamp L1 gate: adopt fast only on measured no-worse accuracy).
GELU_FAST = 0

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


def _div_up(a, b):
    return -(-a // b)


def _footprint_bytes(block_h, core_n_tiles_max, tile_bytes):
    fixed = (K_EXT_TILES * core_n_tiles_max + 1) * tile_bytes
    per_row = (ACT_DEPTH * K_EXT_TILES + OUT_DEPTH * BLOCK_W) * tile_bytes
    return fixed + block_h * per_row


def select_blocking(input_tensor, compute_kernel_config):
    grid = input_tensor.device().compute_with_storage_grid_size()
    batch, _, seq, _ = list(input_tensor.shape)
    tensor_m_tiles = batch * _div_up(seq, TILE)
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)

    assert N_TILES % BLOCK_W == 0
    n_blocks = N_TILES // BLOCK_W
    core_n_blocks = _div_up(n_blocks, grid.x)
    core_n_tiles_max = BLOCK_W * core_n_blocks
    active_cols = _div_up(n_blocks, core_n_blocks)

    core_m_tiles = _div_up(tensor_m_tiles, grid.y)
    active_rows = _div_up(tensor_m_tiles, core_m_tiles)

    fixed = _footprint_bytes(0, core_n_tiles_max, tile_bytes)
    per_row = _footprint_bytes(1, core_n_tiles_max, tile_bytes) - fixed
    max_block_h = (CB_BUDGET_BYTES - fixed) // per_row
    if max_block_h < 1:
        raise RuntimeError(
            f"linear_gelu: W slice of {core_n_tiles_max} tiles on a {grid.x}x{grid.y} grid does not fit "
            f"{CB_BUDGET_BYTES} B of CBs"
        )
    block_h = min(core_m_tiles, max_block_h)
    # fp32 DEST halves the half-sync DEST capacity.
    dest_tiles = 4 if compute_kernel_config.fp32_dest_acc_en else 8
    return Blocking(
        tensor_m_tiles, core_m_tiles, active_rows, core_n_blocks, core_n_tiles_max, active_cols, block_h, dest_tiles
    )


def _subblock(rows, dest_tiles):
    """Largest-area (sb_h, sb_w) with sb_h | rows, sb_w | BLOCK_W, area <= dest_tiles; larger sb_h on ties."""
    best = (1, 1)
    for sb_h in range(1, rows + 1):
        if rows % sb_h:
            continue
        for sb_w in range(1, BLOCK_W + 1):
            if BLOCK_W % sb_w or sb_h * sb_w > dest_tiles:
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
    blocking = select_blocking(input_tensor, compute_kernel_config)
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    block_h = blocking.block_h

    rect = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(blocking.active_cols - 1, blocking.active_rows - 1))
    cores = ttnn.CoreRangeSet([rect])

    cbs = [
        _cb(CB_ACTIVATION_BLOCK, ACT_DEPTH * block_h * K_EXT_TILES, tile_bytes, cores),
        _cb(CB_WEIGHT_RESIDENT, K_EXT_TILES * blocking.core_n_tiles_max, tile_bytes, cores),
        _cb(CB_OUTPUT_TILES, OUT_DEPTH * block_h * BLOCK_W, tile_bytes, cores),
        _cb(CB_BIAS_ONES_TILE, 1, tile_bytes, cores),
    ]

    reader_ct = [
        CB_ACTIVATION_BLOCK,
        CB_WEIGHT_RESIDENT,
        CB_BIAS_ONES_TILE,
        block_h,
        BLOCK_W,
        K_TILES,
        K_EXT_TILES,
        N_TILES,
        tile_bytes,
    ]
    for tensor in (input_tensor, weight, bias):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    seq = list(input_tensor.shape)[-2]
    writer_ct = [CB_OUTPUT_TILES, block_h, BLOCK_W, N_TILES, tile_bytes, _div_up(seq, TILE), seq % TILE]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    compute_ct = [CB_ACTIVATION_BLOCK, CB_WEIGHT_RESIDENT, CB_OUTPUT_TILES, block_h, BLOCK_W, K_EXT_TILES, GELU_FAST]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    core_n_tiles = blocking.core_n_tiles_max
    sb_h, sb_w = _subblock(block_h, blocking.dest_tiles)
    for row in range(blocking.active_rows):
        row_start = row * blocking.core_m_tiles
        rows = min(blocking.core_m_tiles, blocking.tensor_m_tiles - row_start)
        last_block_h = rows - (_div_up(rows, block_h) - 1) * block_h
        tail_sb_h, tail_sb_w = _subblock(last_block_h, blocking.dest_tiles)
        for col in range(blocking.active_cols):
            n_tile_start = col * core_n_tiles
            num_n_blocks = min(core_n_tiles, N_TILES - n_tile_start) // BLOCK_W
            reader_rt[col][row] = [
                input_tensor.buffer_address(),
                weight.buffer_address(),
                bias.buffer_address(),
                row_start,
                rows,
                n_tile_start,
                num_n_blocks,
                row,
            ]
            writer_rt[col][row] = [
                output.buffer_address(),
                row_start,
                rows,
                n_tile_start,
                num_n_blocks,
                sb_h,
                sb_w,
                tail_sb_h,
                tail_sb_w,
            ]
            compute_rt[col][row] = [rows, num_n_blocks, sb_h, sb_w, tail_sb_h, tail_sb_w]

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
    # One group per grid column (ColumnMajor order, group = the column's active rows); row 0 sends.
    w_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
        cores,
        blocking.active_rows,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.ColumnMajor,
    )
    w_mcast.attach(descriptor, W_MCAST_PREFIX, [reader], 0)
    # R2: x row-blocks are shared by a grid row; one group per row, the sender rotates with the row-block.
    x_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
        cores,
        blocking.active_cols,
        ttnn.McastRotatingSenderConfig(),
        ttnn.McastCoreOrder.RowMajor,
    )
    x_mcast.attach(descriptor, X_MCAST_PREFIX, [reader], w_mcast.next_semaphore_id())
    descriptor.kernels = [reader, writer, compute]
    return descriptor
