# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3 program descriptor: blocking (single source of every block knob), CBs, kernels, args.

Regime R0 (interleaved-slab): output pixels M are split over the grid in contiguous tile ranges;
each core runs `num_m_blocks_this_core` M-blocks of `block_m_tiles` tile-rows x all C_out, walking
K in `num_k_blocks` blocks of `block_k_tiles` that never straddle a kernel row (kh).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
TILE_BYTES_BF16 = 2 * TILE * TILE
TILE_BYTES_FP32 = 4 * TILE * TILE
BF16_BYTES = 2

# fp32 DEST, half sync: 4 tiles per DEST half. Bounds the matmul subblock, never a block.
DEST_TILES_FP32_HALF_SYNC = 4

# L1 left free for L1-resident tensors other code keeps alive while this program runs.
L1_TENSOR_HEADROOM_BYTES = 128 * 1024

# CB indices (semantic names; slots are arbitrary but fixed).
CB_HALO_SLAB = 0
CB_IM2COL_RM = 1
CB_IM2COL_TILES = 2
CB_WEIGHTS = 3
CB_BIAS = 4
CB_MATMUL_PARTIALS = 5
CB_OUTPUT_TILES = 6

# Buffer-depth knobs (Blocking Model depth table).
IM2COL_RM_DEPTH_ROWS = 2
WEIGHT_DEPTH = 2
OUT_DEPTH_ROW_GROUPS = 2


def l1_cb_budget_bytes():
    return ttnn.get_max_worker_l1_unreserved_size() - L1_TENSOR_HEADROOM_BYTES


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Conv3x3Geometry:
    input_height: int
    input_width: int
    c_in_p: int
    c_out: int
    padding: int
    stride: int
    dilation: int
    output_height: int
    output_width: int
    in0_tile_bytes: int
    weight_tile_bytes: int
    out_tile_bytes: int
    has_bias: bool

    @property
    def num_out_px(self):
        return self.output_height * self.output_width

    @property
    def tensor_m_tiles(self):
        return _div_up(self.num_out_px, TILE)

    @property
    def tensor_n_tiles(self):
        return _div_up(self.c_out, TILE)

    @property
    def tensor_kkh_tiles(self):
        return _div_up(3 * self.c_in_p, TILE)

    @property
    def tensor_k_tiles(self):
        return 3 * self.tensor_kkh_tiles


@dataclass(frozen=True)
class Conv3x3Blocking:
    grid_x: int
    core_m_tiles: int
    num_m_groups: int
    block_m_tiles: int
    block_k_tiles: int
    block_n_tiles: int
    num_k_blocks: int
    max_m_blocks_per_core: int
    band_len_px: int
    band_stride_px: int
    out_subblock_h: int
    out_subblock_w: int
    im2col_rm_depth_rows: int
    weight_depth: int
    out_depth_row_groups: int
    footprint_bytes: int

    @property
    def slab_px(self):
        return 2 * self.band_stride_px + self.band_len_px


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _out_subblock(block_m_tiles, block_n_tiles):
    """Width first (it is what pays), then the tallest height that still fits DEST."""
    sb_w = _largest_divisor_at_most(block_n_tiles, DEST_TILES_FP32_HALF_SYNC)
    sb_h = _largest_divisor_at_most(block_m_tiles, DEST_TILES_FP32_HALF_SYNC // sb_w)
    return sb_h, sb_w


def _core_m_ranges(tensor_m_tiles, core_m_tiles):
    return [(start, min(core_m_tiles, tensor_m_tiles - start)) for start in range(0, tensor_m_tiles, core_m_tiles)]


def _block_band_len(g: Conv3x3Geometry, o_first, o_last):
    """Pixels one kernel row of the block touches: from tap (kh, 0) of o_first to tap (kh, 2) of o_last."""
    ho0, wo0 = divmod(o_first, g.output_width)
    ho1, wo1 = divmod(o_last, g.output_width)
    return (ho1 - ho0) * g.stride * g.input_width + (wo1 - wo0) * g.stride + 2 * g.dilation + 1


def _max_band_len(g: Conv3x3Geometry, core_m_tiles, block_m_tiles):
    worst = 0
    for start, actual in _core_m_ranges(g.tensor_m_tiles, core_m_tiles):
        for block_start in range(0, actual, block_m_tiles):
            tiles = min(block_m_tiles, actual - block_start)
            o_first = (start + block_start) * TILE
            o_last = min((start + block_start + tiles) * TILE, g.num_out_px) - 1
            worst = max(worst, _block_band_len(g, o_first, o_last))
    return worst


def _footprint(g: Conv3x3Geometry, core_m_tiles, bm, bk, bn):
    num_k_blocks = g.tensor_k_tiles // bk
    max_m_blocks = _div_up(core_m_tiles, bm)
    weight_depth = 1 if num_k_blocks * max_m_blocks == 1 else WEIGHT_DEPTH
    band_len = _max_band_len(g, core_m_tiles, bm)
    band_stride = min(g.dilation * g.input_width, band_len)
    sb_h, sb_w = _out_subblock(bm, bn)
    total = (
        (2 * band_stride + band_len) * g.c_in_p * BF16_BYTES
        + IM2COL_RM_DEPTH_ROWS * bk * TILE_BYTES_BF16
        + bm * bk * g.in0_tile_bytes
        + weight_depth * bk * bn * g.weight_tile_bytes
        + (bn * g.weight_tile_bytes if g.has_bias else 0)
        + bm * bn * TILE_BYTES_FP32
        + OUT_DEPTH_ROW_GROUPS * sb_h * bn * g.out_tile_bytes
    )
    return dict(
        num_k_blocks=num_k_blocks,
        max_m_blocks_per_core=max_m_blocks,
        weight_depth=weight_depth,
        band_len_px=band_len,
        band_stride_px=band_stride,
        out_subblock_h=sb_h,
        out_subblock_w=sb_w,
        footprint_bytes=total,
    )


@functools.lru_cache(maxsize=256)
def select_blocking(g: Conv3x3Geometry, grid, l1_budget) -> Conv3x3Blocking:
    """The one place block extents are chosen. Order of the L1 solve (op_design Mechanism caps #1):
    shrink bk to divisors of the kh block down to one pixel's channels (costs no bytes), then shrink bm
    (re-streams weights and halo), then bk below the tap floor. bn is never shrunk."""
    grid_x, grid_y = grid
    grid_cores = grid_x * grid_y
    core_m_tiles = _div_up(g.tensor_m_tiles, grid_cores)
    num_m_groups = _div_up(g.tensor_m_tiles, core_m_tiles)
    bn = g.tensor_n_tiles
    kkh = g.tensor_kkh_tiles

    divisors = [d for d in range(kkh, 0, -1) if kkh % d == 0]
    tap_floor = g.c_in_p // TILE if g.c_in_p % TILE == 0 else kkh
    bk_above_floor = [d for d in divisors if d >= tap_floor]
    bk_below_floor = [d for d in divisors if d < tap_floor]
    bm_candidates = sorted({_div_up(core_m_tiles, n) for n in range(1, core_m_tiles + 1)}, reverse=True)

    search = [(core_m_tiles, bk) for bk in bk_above_floor]
    search += [(bm, bk_above_floor[-1]) for bm in bm_candidates[1:]]
    search += [(bm_candidates[-1], bk) for bk in bk_below_floor]

    for bm, bk in search:
        fit = _footprint(g, core_m_tiles, bm, bk, bn)
        if fit["footprint_bytes"] <= l1_budget:
            return Conv3x3Blocking(
                grid_x=grid_x,
                core_m_tiles=core_m_tiles,
                num_m_groups=num_m_groups,
                block_m_tiles=bm,
                block_k_tiles=bk,
                block_n_tiles=bn,
                im2col_rm_depth_rows=IM2COL_RM_DEPTH_ROWS,
                out_depth_row_groups=OUT_DEPTH_ROW_GROUPS,
                **fit,
            )
    raise ValueError(
        f"conv2d_3x3: no blocking of {g.input_height}x{g.input_width}x{g.c_in_p}->{g.c_out} "
        f"fits the {l1_budget} B L1 CB budget"
    )


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def _core_coords(num_cores, grid_x):
    """Cores filled row-major over the live grid."""
    return [ttnn.CoreCoord(i % grid_x, i // grid_x) for i in range(num_cores)]


def _core_range_set(num_cores, grid_x):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index, data_format, page_size, num_pages, cores):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor,
    weight_tensor,
    bias_tensor,
    output_tensor,
    *,
    geometry: Conv3x3Geometry,
    blocking: Conv3x3Blocking,
    compute_kernel_config,
) -> ttnn.ProgramDescriptor:
    g, b = geometry, blocking
    has_bias = bias_tensor is not None
    pixel_bytes = g.c_in_p * BF16_BYTES
    if input_tensor.buffer_aligned_page_size() != pixel_bytes:
        raise ValueError(
            f"conv2d_3x3: input page stride {input_tensor.buffer_aligned_page_size()} B != slab row {pixel_bytes} B"
        )

    num_cores = b.num_m_groups
    cores = _core_range_set(num_cores, b.grid_x)
    bm, bk, bn = b.block_m_tiles, b.block_k_tiles, b.block_n_tiles

    cbs = [
        _cb(CB_HALO_SLAB, ttnn.bfloat16, pixel_bytes, b.slab_px, cores),
        _cb(CB_IM2COL_RM, ttnn.bfloat16, TILE_BYTES_BF16, b.im2col_rm_depth_rows * bk, cores),
        _cb(CB_IM2COL_TILES, input_tensor.dtype, g.in0_tile_bytes, bm * bk, cores),
        _cb(CB_WEIGHTS, weight_tensor.dtype, g.weight_tile_bytes, b.weight_depth * bk * bn, cores),
        _cb(CB_MATMUL_PARTIALS, ttnn.float32, TILE_BYTES_FP32, bm * bn, cores),
        _cb(
            CB_OUTPUT_TILES,
            output_tensor.dtype,
            g.out_tile_bytes,
            b.out_depth_row_groups * b.out_subblock_h * bn,
            cores,
        ),
    ]
    if has_bias:
        cbs.append(_cb(CB_BIAS, bias_tensor.dtype, g.weight_tile_bytes, bn, cores))

    reader_ct_args = [
        CB_HALO_SLAB,
        CB_IM2COL_RM,
        g.input_height,
        g.input_width,
        g.output_width,
        g.num_out_px,
        g.c_in_p,
        g.padding,
        g.stride,
        g.dilation,
        bm,
        bk,
        g.tensor_kkh_tiles,
        b.num_k_blocks,
        b.band_stride_px,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    compute_ct_args = [
        CB_IM2COL_RM,
        CB_IM2COL_TILES,
        CB_WEIGHTS,
        CB_BIAS,
        CB_MATMUL_PARTIALS,
        CB_OUTPUT_TILES,
        bm,
        bk,
        bn,
        b.num_k_blocks,
        b.out_subblock_h,
        b.out_subblock_w,
        int(has_bias),
    ]

    writer_ct_args = [
        CB_WEIGHTS,
        CB_BIAS,
        CB_OUTPUT_TILES,
        bm,
        bk,
        bn,
        b.num_k_blocks,
        b.out_subblock_h,
        g.tensor_n_tiles,
        int(has_bias),
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(weight_tensor).get_compile_time_args())
    writer_ct_args.extend(
        ttnn.TensorAccessorArgs(bias_tensor).get_compile_time_args()
        if has_bias
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    input_addr = input_tensor.buffer_address()
    weight_addr = weight_tensor.buffer_address()
    bias_addr = bias_tensor.buffer_address() if has_bias else 0
    output_addr = output_tensor.buffer_address()
    n_tile_start = 0
    for core, (m_tile_start, actual_m_tiles) in zip(
        _core_coords(num_cores, b.grid_x), _core_m_ranges(g.tensor_m_tiles, b.core_m_tiles)
    ):
        num_m_blocks_this_core = _div_up(actual_m_tiles, bm)
        reader_rt[core.x][core.y] = [input_addr, m_tile_start, actual_m_tiles, num_m_blocks_this_core]
        compute_rt[core.x][core.y] = [num_m_blocks_this_core]
        writer_rt[core.x][core.y] = [
            weight_addr,
            bias_addr,
            output_addr,
            m_tile_start,
            actual_m_tiles,
            num_m_blocks_this_core,
            n_tile_start,
        ]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_3x3_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_3x3_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "conv2d_3x3_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_kernel_config.math_fidelity,
            math_approx_mode=compute_kernel_config.math_approx_mode,
            fp32_dest_acc_en=True,
        ),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
