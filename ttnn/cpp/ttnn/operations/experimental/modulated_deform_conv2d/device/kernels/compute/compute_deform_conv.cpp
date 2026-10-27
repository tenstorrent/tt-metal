// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the modulated deformable conv: the sampling geometry, the bilinear columns and the
// conv's matmul in one kernel, so the columns never leave L1.
//
// Kernel positions run outermost, the core's blocks of 32 output rows inside them; an input wider
// than one chunk of channels takes a pass per chunk at each position. During the first pass each
// block's geometry is computed on the SFPU two blocks ahead of its corners, so a reader gathers one
// block while compute works through another, and handed to the reader that owns the block (even
// blocks to the first, odd to the second), which keeps it for every pass. For each (pass, block)
// the reader's four corners are tilized on the unpacker, multiplied by their scalar columns and
// summed in L1, which gives the block's columns of that pass's chunk; a matmul with the pass's
// chunk x C_out slice of the weight then accumulates into the block's output tiles, in fp32 dest
// within the pass and in bfloat16 L1 across passes. After the last pass the bias is added, the ReLU
// applied, and the bfloat16 output handed to the block's reader.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/relu.h"

constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t channel_tiles = get_compile_time_arg_val(3);  // input channels per pass / 32
constexpr uint32_t kernel_positions = get_compile_time_arg_val(4);
constexpr uint32_t tiled_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t geom_in_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t geom_index_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t geom_weight_cb_index = get_compile_time_arg_val(8);
constexpr uint32_t geom_kernel_cb_index = get_compile_time_arg_val(9);
// fp32 bits of the input height and width, and of each less one.
constexpr uint32_t IN_H_FP32 = get_compile_time_arg_val(10);
constexpr uint32_t IN_H_LESS_ONE_FP32 = get_compile_time_arg_val(11);
constexpr uint32_t IN_W_FP32 = get_compile_time_arg_val(12);
constexpr uint32_t IN_W_LESS_ONE_FP32 = get_compile_time_arg_val(13);
// The second reader's buffers, for the odd blocks; the ones above serve the even ones.
constexpr uint32_t odd_corner_block_cb_index = get_compile_time_arg_val(14);
constexpr uint32_t odd_scalar_cb_index = get_compile_time_arg_val(15);
constexpr uint32_t odd_geom_index_cb_index = get_compile_time_arg_val(16);
constexpr uint32_t odd_geom_weight_cb_index = get_compile_time_arg_val(17);
constexpr uint32_t odd_output_cb_index = get_compile_time_arg_val(18);
// The block's columns of one kernel position, the weight slices, the output accumulators and the
// bias.
constexpr uint32_t column_cb_index = get_compile_time_arg_val(19);
constexpr uint32_t conv_weight_cb_index = get_compile_time_arg_val(20);
constexpr uint32_t acc_cb_index = get_compile_time_arg_val(21);
constexpr uint32_t bias_cb_index = get_compile_time_arg_val(22);
constexpr uint32_t OUT_TILES = get_compile_time_arg_val(23);  // C_out / 32
constexpr bool HAS_BIAS = get_compile_time_arg_val(24);
constexpr bool RELU = get_compile_time_arg_val(25);
constexpr uint32_t CHUNKS = get_compile_time_arg_val(26);  // passes per kernel position
constexpr uint32_t CORNERS = 4;
constexpr uint32_t BLOCK_TILES = CORNERS * channel_tiles;
constexpr uint32_t WEIGHT_SLICE_TILES = channel_tiles * OUT_TILES;
// Output tiles per dest pass: full-sync dest holds eight fp32 tiles.
constexpr uint32_t OUT_CHUNK = 8;

// Tiles of geom_in_cb, per block, and of geom_kernel_cb, once, in the reader's order.
constexpr uint32_t IN_DY = 0;
constexpr uint32_t IN_DX = 1;
constexpr uint32_t IN_MASK = 2;  // the logits
constexpr uint32_t IN_ORIGIN_Y = 3;
constexpr uint32_t IN_ORIGIN_X = 4;
constexpr uint32_t GEOM_IN_TILES = 5;
constexpr uint32_t KERNEL_Y = 0;
constexpr uint32_t KERNEL_X = 1;

constexpr uint32_t FP32_ZERO = 0x00000000u;
constexpr uint32_t FP32_ONE = 0x3F800000u;
constexpr uint32_t FP32_MINUS_ONE = 0xBF800000u;

namespace {

inline void copy_values(uint32_t from, uint32_t to) {
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(from, to);
}

// dst = (lo <= dst < hi), as 1.0 or 0.0; scratch is overwritten.
inline void in_range(uint32_t dst, uint32_t scratch, uint32_t lo_fp32, uint32_t hi_fp32) {
    copy_values(dst, scratch);
    unary_ge_tile_init();
    unary_ge_tile(dst, lo_fp32);
    unary_lt_tile_init();
    unary_lt_tile(scratch, hi_fp32);
    mul_binary_tile_init();
    mul_binary_tile(dst, scratch, dst);
}

// Sample point (y, x) = origin + kernel offset + offset; (y0, x0) = floor; ly = y - y0 and
// lx = x - x0. Per axis the near weight is (1 - l) and the far one l, each zeroed where its
// corner row or column lies outside the input; a corner's weight is the product of its two
// axis weights and the mask. Eight dest tiles are live at once, which full-sync dest holds
// in fp32.
void compute_geometry(CircularBuffer& geom_in_cb, uint32_t index_cb_index, uint32_t weight_cb_index) {
    CircularBuffer index_cb(index_cb_index);
    CircularBuffer weight_cb(weight_cb_index);
    geom_in_cb.wait_front(GEOM_IN_TILES);
    reconfig_data_format_srca(geom_in_cb_index);
    copy_init(geom_in_cb_index);

    tile_regs_acquire();
    copy_tile(geom_in_cb_index, IN_DY, 0);
    copy_tile(geom_in_cb_index, IN_ORIGIN_Y, 1);
    copy_tile(geom_in_cb_index, IN_DX, 3);
    copy_tile(geom_in_cb_index, IN_ORIGIN_X, 4);
    copy_tile(geom_in_cb_index, IN_MASK, 6);
    copy_tile(geom_kernel_cb_index, KERNEL_Y, 2);
    copy_tile(geom_kernel_cb_index, KERNEL_X, 5);

    sigmoid_tile_init();
    sigmoid_tile(6);  // the mask, from its logits

    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);  // y
    add_binary_tile(3, 4, 3);
    add_binary_tile(3, 5, 3);  // x
    copy_values(0, 1);
    copy_values(3, 4);
    rounding_op_tile_init();
    floor_tile(1);  // y0
    floor_tile(4);  // x0
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);  // ly
    sub_binary_tile(3, 4, 3);  // lx

    // The top-left corner's page in its image.
    copy_values(1, 2);
    binop_with_scalar_tile_init();
    mul_unary_tile(2, IN_W_FP32);
    add_binary_tile_init();
    add_binary_tile(2, 4, 2);

    // Rows: y0 inside the input weighs (1 - ly), y0 + 1 inside weighs ly.
    copy_values(1, 5);
    in_range(5, 7, FP32_ZERO, IN_H_FP32);
    copy_values(0, 7);
    binop_with_scalar_tile_init();
    rsub_unary_tile(7, FP32_ONE);
    mul_binary_tile_init();
    mul_binary_tile(7, 5, 7);  // near row weight
    in_range(1, 5, FP32_MINUS_ONE, IN_H_LESS_ONE_FP32);
    mul_binary_tile(0, 1, 0);  // far row weight

    // Columns, the same way.
    copy_values(4, 5);
    in_range(5, 1, FP32_ZERO, IN_W_FP32);
    copy_values(3, 1);
    binop_with_scalar_tile_init();
    rsub_unary_tile(1, FP32_ONE);
    mul_binary_tile_init();
    mul_binary_tile(1, 5, 1);  // near column weight
    in_range(4, 5, FP32_MINUS_ONE, IN_W_LESS_ONE_FP32);
    mul_binary_tile(3, 4, 3);  // far column weight

    mul_binary_tile(7, 6, 7);  // near row * mask
    mul_binary_tile(0, 6, 0);  // far row * mask
    mul_binary_tile(7, 1, 4);  // corner 0
    mul_binary_tile(7, 3, 7);  // corner 1
    mul_binary_tile(0, 1, 1);  // corner 2
    mul_binary_tile(0, 3, 0);  // corner 3

    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(2);
    tile_regs_commit();
    geom_in_cb.pop_front(GEOM_IN_TILES);

    index_cb.reserve_back(1);
    weight_cb.reserve_back(CORNERS);
    tile_regs_wait();
    pack_reconfig_data_format(index_cb_index);
    pack_tile(2, index_cb_index);
    pack_reconfig_data_format(weight_cb_index);
    pack_tile(4, weight_cb_index);
    pack_tile(7, weight_cb_index);
    pack_tile(1, weight_cb_index);
    pack_tile(0, weight_cb_index);
    tile_regs_release();
    index_cb.push_back(1);
    weight_cb.push_back(CORNERS);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(corner_block_cb_index, scalar_cb_index, column_cb_index);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    CircularBuffer tiled_cb(tiled_cb_index);
    CircularBuffer geom_in_cb(geom_in_cb_index);
    CircularBuffer column_cb(column_cb_index);
    CircularBuffer conv_weight_cb(conv_weight_cb_index);
    CircularBuffer acc_cb(acc_cb_index);
    // Written once by the reader and kept for every block.
    CircularBuffer(geom_kernel_cb_index).wait_front(2);
    // Every block's output tiles stay reserved while the positions accumulate into them.
    acc_cb.reserve_back(num_blocks * OUT_TILES);

    auto geometry_of = [&](uint32_t block) {
        const bool odd = block & 1;
        compute_geometry(
            geom_in_cb,
            odd ? odd_geom_index_cb_index : geom_index_cb_index,
            odd ? odd_geom_weight_cb_index : geom_weight_cb_index);
    };
    for (uint32_t block = 0; block < num_blocks && block < 2; ++block) {
        geometry_of(block);
    }

    for (uint32_t pass = 0; pass < kernel_positions * CHUNKS; ++pass) {
        conv_weight_cb.wait_front(WEIGHT_SLICE_TILES);
        for (uint32_t block = 0; block < num_blocks; ++block) {
            const bool odd = block & 1;
            const uint32_t corner_cb_index = odd ? odd_corner_block_cb_index : corner_block_cb_index;
            const uint32_t scalar_index = odd ? odd_scalar_cb_index : scalar_cb_index;
            CircularBuffer corner_block_cb(corner_cb_index);
            CircularBuffer scalar_cb(scalar_index);

            // The block's columns of this pass: the four corners times their scalars, summed.
            reconfig_data_format_srca(corner_cb_index);
            pack_reconfig_data_format(tiled_cb_index);
            pack_reconfig_l1_acc(0);
            tilize_init(corner_cb_index, BLOCK_TILES, tiled_cb_index);
            corner_block_cb.wait_front(BLOCK_TILES);
            tiled_cb.reserve_back(BLOCK_TILES);
            tilize_block(corner_cb_index, BLOCK_TILES, tiled_cb_index);
            tiled_cb.push_back(BLOCK_TILES);
            corner_block_cb.pop_front(BLOCK_TILES);
            tilize_uninit(corner_cb_index, tiled_cb_index);

            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(tiled_cb_index, scalar_index);
            tiled_cb.wait_front(BLOCK_TILES);
            // The four corners accumulate into these tiles in L1, so they stay reserved throughout.
            column_cb.reserve_back(channel_tiles);
            for (uint32_t corner = 0; corner < CORNERS; ++corner) {
                pack_reconfig_l1_acc(corner == 0 ? 0 : 1);
                scalar_cb.wait_front(1);
                for (uint32_t t = 0; t < channel_tiles; ++t) {
                    tile_regs_acquire();
                    mul_tiles_bcast<BroadcastType::COL>(tiled_cb_index, scalar_index, corner * channel_tiles + t, 0, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    // Packs to slot t explicitly; the default would advance the write
                    // pointer past the reserved tiles after the first corner.
                    pack_tile<true>(0, column_cb_index, t);
                    tile_regs_release();
                }
                scalar_cb.pop_front(1);
            }
            pack_reconfig_l1_acc(0);
            column_cb.push_back(channel_tiles);
            tiled_cb.pop_front(BLOCK_TILES);

            // The block's output += its columns times the pass's weight slice: fp32 in dest over the
            // slice's rows, and in L1 across passes; the first pass overwrites.
            column_cb.wait_front(channel_tiles);
            reconfig_data_format(conv_weight_cb_index, column_cb_index);
            pack_reconfig_data_format(acc_cb_index);
            for (uint32_t n0 = 0; n0 < OUT_TILES; n0 += OUT_CHUNK) {
                const uint32_t chunk = OUT_TILES - n0 < OUT_CHUNK ? OUT_TILES - n0 : OUT_CHUNK;
                matmul_block_init(column_cb_index, conv_weight_cb_index, 0, chunk, 1, channel_tiles);
                tile_regs_acquire();
                for (uint32_t k = 0; k < channel_tiles; ++k) {
                    matmul_block(
                        column_cb_index, conv_weight_cb_index, k, k * OUT_TILES + n0, 0, 0, chunk, 1, channel_tiles);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_l1_acc(pass == 0 ? 0 : 1);
                for (uint32_t i = 0; i < chunk; ++i) {
                    pack_tile<true>(i, acc_cb_index, block * OUT_TILES + n0 + i);
                }
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);
            column_cb.pop_front(channel_tiles);
            if (pass == 0 && block + 2 < num_blocks) {
                geometry_of(block + 2);
            }
        }
        conv_weight_cb.pop_front(WEIGHT_SLICE_TILES);
    }
    acc_cb.push_back(num_blocks * OUT_TILES);

    // Bias, ReLU, bfloat16: each block's output to the reader that owns the block.
    acc_cb.wait_front(num_blocks * OUT_TILES);
    if constexpr (HAS_BIAS) {
        CircularBuffer(bias_cb_index).wait_front(OUT_TILES);
        reconfig_data_format(acc_cb_index, bias_cb_index);
    } else {
        reconfig_data_format_srca(acc_cb_index);
    }
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t out_index = (block & 1) ? odd_output_cb_index : output_cb_index;
        CircularBuffer output_cb(out_index);
        output_cb.reserve_back(OUT_TILES);
        pack_reconfig_data_format(out_index);
        for (uint32_t n0 = 0; n0 < OUT_TILES; n0 += OUT_CHUNK) {
            const uint32_t chunk = OUT_TILES - n0 < OUT_CHUNK ? OUT_TILES - n0 : OUT_CHUNK;
            if constexpr (HAS_BIAS) {
                add_bcast_rows_init(acc_cb_index, bias_cb_index);
            } else {
                copy_init(acc_cb_index);
            }
            tile_regs_acquire();
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tile = block * OUT_TILES + n0 + i;
                if constexpr (HAS_BIAS) {
                    add_tiles_bcast_rows(acc_cb_index, bias_cb_index, tile, n0 + i, i);
                } else {
                    copy_tile(acc_cb_index, tile, i);
                }
            }
            if constexpr (RELU) {
                relu_tile_init();
                for (uint32_t i = 0; i < chunk; ++i) {
                    relu_tile(i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < chunk; ++i) {
                pack_tile<true>(i, out_index, n0 + i);
            }
            tile_regs_release();
        }
        output_cb.push_back(OUT_TILES);
    }
}
