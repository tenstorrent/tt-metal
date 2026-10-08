// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute of the modulated deformable im2col. For each (block, kernel position)
// the reader lands the four corners as one row-major block, 32 rows of
// [corner 0 | corner 1 | corner 2 | corner 3], plus one scalar tile per corner.
// This kernel tilizes the block on the unpacker, then multiplies each corner's
// tiles by its scalar column and sums the four in L1:
//
//   out[row, col] = sum_corner corner[row, col] * scalar[row, 0]
//
// which is the masked bilinear sample of the block's rows at that position.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t corner_block_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(2);
constexpr uint32_t channel_tiles = get_compile_time_arg_val(3);
constexpr uint32_t kernel_positions = get_compile_time_arg_val(4);
constexpr uint32_t tiled_cb_index = get_compile_time_arg_val(5);
constexpr uint32_t CORNERS = 4;
constexpr uint32_t BLOCK_TILES = CORNERS * channel_tiles;

void kernel_main() {
    compute_kernel_hw_startup(corner_block_cb_index, scalar_cb_index, output_cb_index);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    CircularBuffer corner_block_cb(corner_block_cb_index);
    CircularBuffer tiled_cb(tiled_cb_index);
    CircularBuffer scalar_cb(scalar_cb_index);
    CircularBuffer output_cb(output_cb_index);

    for (uint32_t unit = 0; unit < num_blocks * kernel_positions; ++unit) {
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
                mul_tiles_bcast<BroadcastType::COL>(tiled_cb_index, scalar_cb_index, corner * channel_tiles + t, 0, 0);
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
