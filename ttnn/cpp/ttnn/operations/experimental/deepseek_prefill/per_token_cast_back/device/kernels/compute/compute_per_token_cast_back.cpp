// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_back: out = decode(input_e4m3) * scale, where scale is one fp32
// scalar per token (row) per 128-wide block. A 128-element block spans
// 4 consecutive 32x32 tiles; within any tile the scale is a per-row scalar (constant across columns),
// so we broadcast it with mul_tiles_bcast_cols (BroadcastType::COL = filled column 0, per notes §6).
//
// The reader builds one bcast operand tile in cb_scale_bcast with column 0 = scale[:, block_idx]
// (face-aware). There is no in-kernel column-shift LLK on Blackhole (notes §6), so the per-block
// column selection lives in the reader's data layout, not a shift here.
//
// Per block = tile_h rows x 128 cols = 4 tiles for default 32-wide tiles:
//   Phase 1: decode e4m3 -> fp32 and tilize
//   Phase 2: multiply by broadcast scale and untilize to output

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_input_e4m3_id = get_compile_time_arg_val(0);
    CircularBuffer cb_input_e4m3(cb_input_e4m3_id);
    constexpr uint32_t cb_in_rm_id = get_compile_time_arg_val(1);
    CircularBuffer cb_in_rm(cb_in_rm_id);
    constexpr uint32_t cb_in_tile_id = get_compile_time_arg_val(2);
    CircularBuffer cb_in_tile(cb_in_tile_id);
    constexpr uint32_t cb_scale_bcast_id = get_compile_time_arg_val(3);
    CircularBuffer cb_scale_bcast(cb_scale_bcast_id);
    constexpr uint32_t cb_out_tile_id = get_compile_time_arg_val(4);
    CircularBuffer cb_out_tile(cb_out_tile_id);
    constexpr uint32_t cb_out_fp32_id = get_compile_time_arg_val(5);
    CircularBuffer cb_out_fp32(cb_out_fp32_id);
    // Tile dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t block_w = 128;                // BlockW
    constexpr uint32_t block_wt = block_w / tile_w;  // BlockWt
    constexpr uint32_t block_ht = 1;                 // BlockHt
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;

    uint32_t num_blocks = get_arg_val<uint32_t>(0);  // tile_h x 128 blocks for this core

    compute_kernel_hw_startup(cb_input_e4m3_id, cb_out_fp32_id);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            // ----- Phase 1: decode e4m3 -> fp32 and tilize -----
            // tilize cannot decode e4m3 into a Float32 dest, so decode first via copy_tile.
            reconfig_data_format_srca(cb_input_e4m3_id);
            pack_reconfig_data_format(cb_in_rm_id);
            copy_tile_init(cb_input_e4m3_id);
            for (uint32_t s = 0; s < tiles_per_block; ++s) {
                cb_input_e4m3.wait_front(1);
                cb_in_rm.reserve_back(1);
                tile_regs_acquire();
                copy_tile(cb_input_e4m3_id, 0, IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_in_rm_id);
                tile_regs_release();
                cb_in_rm.push_back(1);
                cb_input_e4m3.pop_front(1);
            }

            reconfig_data_format_srca(cb_in_rm_id);
            pack_reconfig_data_format(cb_in_tile_id);
            tilize_init(cb_in_rm_id, tiles_per_block, cb_in_tile_id);
            cb_in_rm.wait_front(tiles_per_block);
            cb_in_tile.reserve_back(tiles_per_block);
            tilize_block(cb_in_rm_id, tiles_per_block, cb_in_tile_id);
            cb_in_tile.push_back(tiles_per_block);
            cb_in_rm.pop_front(tiles_per_block);
            tilize_uninit(cb_in_rm_id, cb_in_tile_id);

            // ----- Phase 2: multiply by broadcast scale and untilize to output -----
            // mul writes all tiles_per_block(=4) tiles into DEST, and pack_untilize_dest untilizes them
            // to the row-major output in the packer, avoiding a separate cb_out_tile round-trip. In 32-bit
            // DEST (fp32_dest_acc), the half-sync pack-untilize block cap is 4 tiles = one 128-wide block.
            reconfig_data_format(cb_in_tile_id, cb_scale_bcast_id);
            mul_bcast_cols_init_short(cb_in_tile_id, cb_scale_bcast_id);
            pack_untilize_dest_init<tiles_per_block, tiles_per_block>(cb_out_fp32_id);
            cb_in_tile.wait_front(tiles_per_block);
            cb_scale_bcast.wait_front(block_ht);
            cb_out_fp32.reserve_back(tiles_per_block);
            tile_regs_acquire();
            for (uint32_t k = 0; k < block_wt; ++k) {
                mul_tiles_bcast_cols(cb_in_tile_id, cb_scale_bcast_id, k, 0, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<tiles_per_block>(cb_out_fp32_id);
            tile_regs_release();
            cb_out_fp32.push_back(tiles_per_block);
            cb_in_tile.pop_front(tiles_per_block);
            cb_scale_bcast.pop_front(block_ht);
            pack_untilize_uninit(cb_out_fp32_id);
        }
    }
}
