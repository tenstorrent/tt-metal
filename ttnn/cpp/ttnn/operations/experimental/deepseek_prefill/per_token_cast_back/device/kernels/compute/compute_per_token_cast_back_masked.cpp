// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Masked compute for per_token_cast_back. Identical block pipeline to the plain compute
// (input_e4m3 RM -> fp32 RM -> tilize -> scale bcast multiply -> untilize), but num_blocks is
// dynamic — it depends on this device's per-expert token counts, which are only known on device — so
// it is read from cb_nblocks (published once by the masked reader) instead of a host runtime arg.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"

void kernel_main() {
    constexpr uint32_t cb_input_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in_rm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_tile = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scale_bcast = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out_tile = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out_fp32 = get_compile_time_arg_val(5);
    constexpr uint32_t cb_nblocks = get_compile_time_arg_val(6);
    constexpr uint32_t tile_h = get_compile_time_arg_val(7);
    constexpr uint32_t tile_w = get_compile_time_arg_val(8);
    constexpr uint32_t block_w = 128;
    constexpr uint32_t block_wt = block_w / tile_w;
    constexpr uint32_t block_ht = 1;
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;

    // Dynamic block count for this core: published by the masked reader into cb_nblocks (1 page).
    cb_wait_front(cb_nblocks, 1);
    uint32_t num_blocks = read_tile_value(cb_nblocks, 0, 0);
    cb_pop_front(cb_nblocks, 1);

    compute_kernel_hw_startup(cb_input_e4m3, cb_out_fp32);

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            // ----- Phase 1: input_e4m3 row-major -> fp32 row-major (one tile at a time, index 0) -----
            reconfig_data_format_srca(cb_input_e4m3);
            pack_reconfig_data_format(cb_in_rm);
            copy_tile_init(cb_input_e4m3);
            for (uint32_t s = 0; s < tiles_per_block; ++s) {
                cb_wait_front(cb_input_e4m3, 1);
                cb_reserve_back(cb_in_rm, 1);
                tile_regs_acquire();
                copy_tile(cb_input_e4m3, 0, IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_in_rm);
                tile_regs_release();
                cb_push_back(cb_in_rm, 1);
                cb_pop_front(cb_input_e4m3, 1);
            }

            // ----- Phase 2a: tilize fp32 input row-major -> tile -----
            reconfig_data_format_srca(cb_in_rm);
            pack_reconfig_data_format(cb_in_tile);
            tilize_init(cb_in_rm, tiles_per_block, cb_in_tile);
            cb_wait_front(cb_in_rm, tiles_per_block);
            cb_reserve_back(cb_in_tile, tiles_per_block);
            tilize_block(cb_in_rm, tiles_per_block, cb_in_tile);
            cb_push_back(cb_in_tile, tiles_per_block);
            cb_pop_front(cb_in_rm, tiles_per_block);
            tilize_uninit(cb_in_rm, cb_in_tile);

            // ----- Phase 2c: block broadcast multiply -----
            reconfig_data_format(cb_in_tile, cb_scale_bcast);
            pack_reconfig_data_format(cb_out_tile);
            mul_bcast_cols_init_short(cb_in_tile, cb_scale_bcast);
            cb_wait_front(cb_in_tile, tiles_per_block);
            cb_wait_front(cb_scale_bcast, block_ht);
            cb_reserve_back(cb_out_tile, tiles_per_block);
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                for (uint32_t k = 0; k < block_wt; ++k) {
                    uint32_t in_idx = block_h_idx * block_wt + k;
                    tile_regs_acquire();
                    mul_tiles_bcast_cols(cb_in_tile, cb_scale_bcast, in_idx, block_h_idx, IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_out_tile);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_out_tile, tiles_per_block);
            cb_pop_front(cb_in_tile, tiles_per_block);
            cb_pop_front(cb_scale_bcast, block_ht);

            // ----- Phase 3: untilize cb_out_tile -> fp32 row-major output -----
            reconfig_data_format_srca(cb_out_tile);
            pack_reconfig_data_format(cb_out_fp32);
            untilize_init(cb_out_tile);
            cb_wait_front(cb_out_tile, tiles_per_block);
            cb_reserve_back(cb_out_fp32, tiles_per_block);
            untilize_block(cb_out_tile, tiles_per_block, cb_out_fp32);
            cb_push_back(cb_out_fp32, tiles_per_block);
            cb_pop_front(cb_out_tile, tiles_per_block);
            untilize_uninit(cb_out_tile);
        }
    }
}
