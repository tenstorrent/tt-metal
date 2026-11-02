# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""qkv_proj_heads_rope ProgramDescriptor: regime R1 (2D token_rows x head_cols output-stationary, unicast reads)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
HIDDEN = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN // NUM_HEADS
NUM_THIRDS = 3
TILES_PER_HEAD = HEAD_DIM // TILE
HIDDEN_TILES = HIDDEN // TILE
# The bias rides as one extra K tile (in0: ones-column tile, in1: bias tile row).
K_TILES = HIDDEN_TILES + 1
TOTAL_HEADS = NUM_THIRDS * NUM_HEADS
TABLE_TILES_PER_ROW = 2 * TILES_PER_HEAD

# Blocking-model host constants (op_design.md, Blocking Model).
K_BLOCK = 5
ACT_DEPTH = 2
WEIGHT_DEPTH = 2
TABLE_DEPTH = 2
OUT_DEPTH = 2
CB_BUDGET_BYTES = 900 * 1024
OCCUPANCY_SLACK = 0.05
HEADS_PER_CORE_CANDIDATES = tuple(h for h in range(1, NUM_HEADS + 1) if NUM_HEADS % h == 0)

BF16_TILE_BYTES = ttnn.tile_size(ttnn.bfloat16)
FP32_TILE_BYTES = ttnn.tile_size(ttnn.float32)

CB_ACTIVATION_BLOCK = 0
CB_WEIGHT_BLOCK = 1
CB_MATMUL_PARTIALS = 2
CB_ROTARY_TABLE = 3
CB_OUT_TILES = 4

assert K_TILES % K_BLOCK == 0, "k_block must divide K' (ragged last K-block not built)"


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Blocking:
    heads_per_core: int
    grid_m: int
    grid_n: int
    block_m: int
    k_block: int
    tensor_row_tiles: int  # Mt
    seq_tiles: int  # St
    sb_h: int
    sb_w: int

    @property
    def n_tiles(self):
        return TILES_PER_HEAD * self.heads_per_core


def _cb_pages(block_m, n_tiles, k_block):
    """Page count per CB: the single source for the L1 footprint and the allocation (l1_ledger.md)."""
    return {
        CB_ACTIVATION_BLOCK: ACT_DEPTH * block_m * k_block,
        CB_WEIGHT_BLOCK: WEIGHT_DEPTH * k_block * n_tiles,
        CB_MATMUL_PARTIALS: block_m * n_tiles,
        CB_ROTARY_TABLE: TABLE_DEPTH * TABLE_TILES_PER_ROW,
        CB_OUT_TILES: OUT_DEPTH * n_tiles,
    }


CB_FORMATS = {
    CB_ACTIVATION_BLOCK: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_WEIGHT_BLOCK: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_MATMUL_PARTIALS: (FP32_TILE_BYTES, ttnn.float32),
    CB_ROTARY_TABLE: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_OUT_TILES: (BF16_TILE_BYTES, ttnn.bfloat16),
}


def _footprint(block_m, n_tiles, k_block):
    return sum(pages * CB_FORMATS[cb][0] for cb, pages in _cb_pages(block_m, n_tiles, k_block).items())


def block_m_cap(n_tiles, k_block=K_BLOCK):
    """Largest block_m whose CB footprint fits CB_BUDGET_BYTES (ledger formula, derived from _cb_pages)."""
    fixed = _footprint(0, n_tiles, k_block)
    per_row = _footprint(1, n_tiles, k_block) - fixed
    return (CB_BUDGET_BYTES - fixed) // per_row


def _dest_tiles(compute_kernel_config):
    tiles = 4 if compute_kernel_config.fp32_dest_acc_en else 8
    return tiles * (2 if compute_kernel_config.dst_full_sync_en else 1)


def _subblock(block_m, n_tiles, dest_tiles):
    """Max sb_h*sb_w <= dest_tiles with sb_h | block_m and sb_w | n_tiles; ties -> wider sb_w."""
    best = (1, 1)
    for sb_w in range(1, n_tiles + 1):
        if n_tiles % sb_w:
            continue
        for sb_h in range(1, block_m + 1):
            if block_m % sb_h or sb_h * sb_w > dest_tiles:
                continue
            if sb_h * sb_w > best[0] * best[1] or (sb_h * sb_w == best[0] * best[1] and sb_w > best[1]):
                best = (sb_h, sb_w)
    return best


def select_blocking(device, batch, seq, compute_kernel_config):
    """Selection function (op_design.md, Core assignment): occupancy first (5% slack), then min ingress."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    seq_tiles = _div_up(seq, TILE)
    mt = batch * seq_tiles

    candidates = []
    for heads_per_core in HEADS_PER_CORE_CANDIDATES:
        grid_n = TOTAL_HEADS // heads_per_core
        grid_m = min(num_cores // grid_n, mt)
        if grid_m == 0:
            continue
        n_tiles = TILES_PER_HEAD * heads_per_core
        cap = block_m_cap(n_tiles)
        if cap < 1:
            continue
        rows_max = _div_up(mt, grid_m)
        block_m = min(rows_max, cap)
        num_blocks = _div_up(rows_max, block_m)
        tiles_max = rows_max * n_tiles
        ingress = K_TILES * (rows_max + num_blocks * n_tiles)
        candidates.append((tiles_max, ingress, heads_per_core, grid_m, grid_n, block_m))
    if not candidates:
        raise RuntimeError("qkv_proj_heads_rope: no core assignment fits the compute grid")

    min_tiles = min(c[0] for c in candidates)
    kept = [c for c in candidates if c[0] <= (1 + OCCUPANCY_SLACK) * min_tiles]
    _, _, heads_per_core, grid_m, grid_n, block_m = min(kept, key=lambda c: (c[1], -c[2]))
    sb_h, sb_w = _subblock(block_m, TILES_PER_HEAD * heads_per_core, _dest_tiles(compute_kernel_config))
    return Blocking(
        heads_per_core=heads_per_core,
        grid_m=grid_m,
        grid_n=grid_n,
        block_m=block_m,
        k_block=K_BLOCK,
        tensor_row_tiles=mt,
        seq_tiles=seq_tiles,
        sb_h=sb_h,
        sb_w=sb_w,
    )


def _cb(index, num_pages, cores):
    page_size, data_format = CB_FORMATS[index]
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _core_assignment(device, blocking):
    """Logical core l = n_group*grid_m + m_group, row-major over the compute grid (an N-group is consecutive)."""
    grid_x = device.compute_with_storage_grid_size().x
    mt, grid_m = blocking.tensor_row_tiles, blocking.grid_m
    base, extra = divmod(mt, grid_m)
    row_ranges = []
    start = 0
    for m_group in range(grid_m):
        rows = base + (1 if m_group < extra else 0)
        row_ranges.append((start, rows))
        start += rows
    assignment = []
    for n_group in range(blocking.grid_n):
        for m_group in range(grid_m):
            logical = n_group * grid_m + m_group
            core = ttnn.CoreCoord(logical % grid_x, logical // grid_x)
            assignment.append((core, n_group, m_group, *row_ranges[m_group]))
    return assignment


def create_program_descriptor(input_tensor, weight, bias, cos, sin, outputs, blocking, compute_kernel_config):
    device = input_tensor.device()
    n_tiles = blocking.n_tiles
    block_m = blocking.block_m
    k_block = blocking.k_block
    num_k_blocks = K_TILES // k_block
    weight_width_tiles = weight.padded_shape[-1] // TILE
    groups_per_third = NUM_HEADS // blocking.heads_per_core

    assignment = _core_assignment(device, blocking)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, *_ in assignment])
    cbs = [_cb(index, pages, cores) for index, pages in _cb_pages(block_m, n_tiles, k_block).items()]

    reader_ct_args = [
        CB_ACTIVATION_BLOCK,
        CB_ROTARY_TABLE,
        block_m,
        k_block,
        num_k_blocks,
        HIDDEN_TILES,
        blocking.seq_tiles,
        BF16_TILE_BYTES,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(sin).get_compile_time_args())

    writer_ct_args = [
        CB_WEIGHT_BLOCK,
        CB_OUT_TILES,
        block_m,
        n_tiles,
        k_block,
        num_k_blocks,
        HIDDEN_TILES,
        weight_width_tiles,
        blocking.seq_tiles,
        NUM_HEADS,
        TILES_PER_HEAD,
        BF16_TILE_BYTES,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(bias).get_compile_time_args())
    for output in outputs:
        writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_ct_args = [
        CB_ACTIVATION_BLOCK,
        CB_WEIGHT_BLOCK,
        CB_MATMUL_PARTIALS,
        CB_ROTARY_TABLE,
        CB_OUT_TILES,
        block_m,
        blocking.heads_per_core,
        TILES_PER_HEAD,
        k_block,
        num_k_blocks,
        blocking.sb_h,
        blocking.sb_w,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    cos_addr, sin_addr = cos.buffer_address(), sin.buffer_address()
    w_addr, b_addr = weight.buffer_address(), bias.buffer_address()
    out_addrs = [o.buffer_address() for o in outputs]
    for core, n_group, m_group, row_start, core_rows in assignment:
        third = n_group // groups_per_third
        first_head = (n_group % groups_per_third) * blocking.heads_per_core
        first_col_tile = third * HIDDEN_TILES + TILES_PER_HEAD * first_head
        is_rotary = int(third < 2)
        reader_rt[core.x][core.y] = [
            x_addr,
            cos_addr,
            sin_addr,
            row_start,
            core_rows,
            is_rotary,
            n_group,
            blocking.grid_n,
        ]
        writer_rt[core.x][core.y] = [
            w_addr,
            b_addr,
            *out_addrs,
            third,
            first_col_tile,
            first_head,
            row_start,
            core_rows,
            m_group,
            blocking.grid_m,
        ]
        compute_rt[core.x][core.y] = [core_rows, is_rotary]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_reader.cpp"),
            core_ranges=cores,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_writer.cpp"),
            core_ranges=cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_compute.cpp"),
            core_ranges=cores,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
