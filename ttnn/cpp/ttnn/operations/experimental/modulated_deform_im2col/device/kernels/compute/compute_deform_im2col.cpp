// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the modulated deformable im2col.
//
// Per block of 32 output rows it first computes the block's sampling geometry on
// the SFPU, in fp32, for all kernel positions at once: the reader hands in one
// tile each of the y and x offsets, the mask, and the kernel window's base
// coordinates, laid out [row, kernel position], and gets back the corners' top
// left (y0, x0) as int32 tiles and the four masked bilinear weights as bfloat16
// tiles, in the same layout. The geometry of the next block is computed before
// this block's corners are combined, so the reader has it as soon as it needs it.
//
// For each (block, kernel position) the reader then lands the four corners as
// one row-major block, 32 rows of [corner 0 | corner 1 | corner 2 | corner 3],
// plus one scalar tile per corner holding that corner's weight in column 0.
// This kernel tilizes the block on the unpacker, then multiplies each corner's
// tiles by its scalar column and sums the four in L1:
//
//   out[row, col] = sum_corner corner[row, col] * scalar[row, 0]
//
// which is the masked bilinear sample of the block's rows at that position.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
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

constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t channel_tiles = get_compile_time_arg_val(3);
constexpr uint32_t kernel_positions = get_compile_time_arg_val(4);
constexpr uint32_t tiled_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t geom_in_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t geom_index_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t geom_weight_cb_index = get_compile_time_arg_val(8);
constexpr uint32_t CORNERS = 4;
constexpr uint32_t BLOCK_TILES = CORNERS * channel_tiles;

// Tiles of geom_in_cb, in the reader's order.
constexpr uint32_t IN_DY = 0;
constexpr uint32_t IN_DX = 1;
constexpr uint32_t IN_MASK = 2;
constexpr uint32_t IN_BASE_Y = 3;
constexpr uint32_t IN_BASE_X = 4;
constexpr uint32_t GEOM_IN_TILES = 5;

constexpr uint32_t FP32_ONE = 0x3F800000u;

namespace {

// y = base_y + dy, x = base_x + dx; (y0, x0) = floor; the weights are the products of
// (1 - ly or ly) and (1 - lx or lx) with the mask, ly = y - y0 and lx = x - x0. All eight
// dest tiles are live at once, which full-sync dest holds in fp32.
void compute_geometry(CircularBuffer& geom_in_cb, CircularBuffer& index_cb, CircularBuffer& weight_cb) {
    geom_in_cb.wait_front(GEOM_IN_TILES);
    reconfig_data_format_srca(geom_in_cb_index);
    copy_init(geom_in_cb_index);

    tile_regs_acquire();
    copy_tile(geom_in_cb_index, IN_DY, 0);
    copy_tile(geom_in_cb_index, IN_BASE_Y, 1);
    copy_tile(geom_in_cb_index, IN_DX, 2);
    copy_tile(geom_in_cb_index, IN_BASE_X, 3);
    copy_tile(geom_in_cb_index, IN_MASK, 4);

    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // y
    add_binary_tile(2, 3, 2);  // x
    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(0, 1);
    copy_dest_values<DataFormat::Float32>(2, 3);
    rounding_op_tile_init();
    floor_tile(1);  // y0
    floor_tile(3);  // x0
    sub_binary_tile_init();
    sub_binary_tile(0, 1, 0);  // ly
    sub_binary_tile(2, 3, 2);  // lx

    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(0, 5);
    copy_dest_values<DataFormat::Float32>(2, 6);
    binop_with_scalar_tile_init();
    rsub_unary_tile(5, FP32_ONE);  // hy
    rsub_unary_tile(6, FP32_ONE);  // hx

    mul_binary_tile_init();
    mul_binary_tile(5, 4, 5);  // hy * mask
    mul_binary_tile(0, 4, 4);  // ly * mask
    mul_binary_tile(5, 6, 7);  // corner 0: hy * hx * mask
    mul_binary_tile(4, 6, 6);  // corner 2: ly * hx * mask
    mul_binary_tile(5, 2, 5);  // corner 1: hy * lx * mask
    mul_binary_tile(4, 2, 4);  // corner 3: ly * lx * mask

    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(1);
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Int32>(3);
    tile_regs_commit();
    geom_in_cb.pop_front(GEOM_IN_TILES);

    index_cb.reserve_back(2);
    weight_cb.reserve_back(CORNERS);
    tile_regs_wait();
    pack_reconfig_data_format(geom_index_cb_index);
    pack_tile(1, geom_index_cb_index);
    pack_tile(3, geom_index_cb_index);
    pack_reconfig_data_format(geom_weight_cb_index);
    pack_tile(7, geom_weight_cb_index);
    pack_tile(5, geom_weight_cb_index);
    pack_tile(6, geom_weight_cb_index);
    pack_tile(4, geom_weight_cb_index);
    tile_regs_release();
    index_cb.push_back(2);
    weight_cb.push_back(CORNERS);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(corner_block_cb_index, scalar_cb_index, output_cb_index);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    CircularBuffer corner_block_cb(corner_block_cb_index);
    CircularBuffer tiled_cb(tiled_cb_index);
    CircularBuffer scalar_cb(scalar_cb_index);
    CircularBuffer output_cb(output_cb_index);
    CircularBuffer geom_in_cb(geom_in_cb_index);
    CircularBuffer index_cb(geom_index_cb_index);
    CircularBuffer weight_cb(geom_weight_cb_index);

    if (num_blocks > 0) {
        compute_geometry(geom_in_cb, index_cb, weight_cb);
    }
    for (uint32_t block = 0; block < num_blocks; ++block) {
        if (block + 1 < num_blocks) {
            compute_geometry(geom_in_cb, index_cb, weight_cb);
        }
        reconfig_data_format_srca(corner_block_cb_index);
        pack_reconfig_data_format(tiled_cb_index);
        for (uint32_t kk = 0; kk < kernel_positions; ++kk) {
            // The packer must overwrite tiled_cb, not accumulate into it.
            pack_reconfig_l1_acc(0);
            tilize_init(corner_block_cb_index, BLOCK_TILES, tiled_cb_index);
            corner_block_cb.wait_front(BLOCK_TILES);
            tiled_cb.reserve_back(BLOCK_TILES);
            tilize_block(corner_block_cb_index, BLOCK_TILES, tiled_cb_index);
            tiled_cb.push_back(BLOCK_TILES);
            corner_block_cb.pop_front(BLOCK_TILES);
            tilize_uninit(corner_block_cb_index, tiled_cb_index);

            bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(tiled_cb_index, scalar_cb_index);
            tiled_cb.wait_front(BLOCK_TILES);
            // The four corners accumulate into these tiles in L1, so they stay reserved throughout.
            output_cb.reserve_back(channel_tiles);
            for (uint32_t corner = 0; corner < CORNERS; ++corner) {
                pack_reconfig_l1_acc(corner == 0 ? 0 : 1);
                scalar_cb.wait_front(1);
                for (uint32_t t = 0; t < channel_tiles; ++t) {
                    tile_regs_acquire();
                    mul_tiles_bcast<BroadcastType::COL>(
                        tiled_cb_index, scalar_cb_index, corner * channel_tiles + t, 0, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    // Packs to slot t explicitly; the default would advance the write
                    // pointer past the reserved tiles after the first corner.
                    pack_tile<true>(0, output_cb_index, t);
                    tile_regs_release();
                }
                scalar_cb.pop_front(1);
            }
            pack_reconfig_l1_acc(0);
            output_cb.push_back(channel_tiles);
            tiled_cb.pop_front(BLOCK_TILES);
        }
    }
}
