// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_to_fp8, step 5: out = cast(input / scale) to e4m3, scale = clamp(amax,1e-4)/448
// per 128-element block.
//
// Per block = tile_h rows x 128 cols = 4 tiles for default 32-wide tiles:
//   1. tilize cb_in -> cb_tile.
//   2. compute per-row amax over the 128-element block, clamp(>=1e-4), multiply by 1/448
//      -> scale (col 0) -> cb_scale_tiles. recip(scale) -> 1/scale -> cb_inv_scale_tiles.
//   3+4. divide: cb_tile * bcast_col(cb_inv_scale_tiles), then pack_untilize_dest the divided
//      tiles straight from DST into cb_output_e4m3 (scaled, cast to e4m3) -- no cb_out_tile
//      L1 round-trip and no separate untilize math pass.
// The writer extracts column 0 of cb_scale_tiles into the scale output [..., M, H/128].
//
// fp32_dest_acc_en=True (required for e4m3 on Blackhole; also gives fp32 reduce/divide precision).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"  // abs_tile / abs_tile_init
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tile = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_abs = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t cb_inv_scale_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_output_e4m3 = get_compile_time_arg_val(6);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(7);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(8);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(9);

    // Tile width from the tensor's tile spec.
    constexpr uint32_t block_w = 128;  // BlockW
    constexpr uint32_t tile_w = get_compile_time_arg_val(10);
    constexpr uint32_t block_wt = block_w / tile_w;              // BlockWt
    constexpr uint32_t block_ht = get_compile_time_arg_val(11);  // BlockHt (tile-rows per block)
    constexpr uint32_t tiles_per_block = block_ht * block_wt;
    // Tiles processed per tile_regs acquire for abs/divide. Must divide tiles_per_block.
    // Set to tiles_per_block (with dst_full_sync_en) to batch the whole block in one acquire,
    // or to block_wt (half-sync) to keep math<->pack double-buffering across tile-rows.
    constexpr uint32_t acq_tiles = get_compile_time_arg_val(12);

    uint32_t num_blocks = get_arg_val<uint32_t>(0);  // block_ht*tile_h x 128 blocks for this core

    // Configure the unpacker hw on the fp32 reduce/abs operand so num_faces / tile dims are full
    // (4 faces). Configuring on a bf16 cb_in instead leaves the fp32 reduce reading only 2 faces
    // (within-tile column reduce sees cols 0-15) for bf16 input; tilize_init re-inits for cb_in.
    compute_kernel_hw_startup(cb_abs, cb_output_e4m3);
    cb_wait_front(cb_scaler, 1);  // reader-filled 1.0 scaler, reused for every reduce

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ----- 1. tilize input row-major -> tile (one tilize_block per tile-row) -----
        // The reader fills cb_in as [block_ht*tile_h x 128] row-major; tilize_block treats each
        // block_wt-tile run as one [tile_h x 128] tile-row, so tile-row r reads/writes tiles
        // [r*block_wt, r*block_wt+block_wt).
        reconfig_data_format_srca(cb_in);
        pack_reconfig_data_format(cb_tile);
        tilize_init(cb_in, block_wt, cb_tile);
        cb_wait_front(cb_in, tiles_per_block);
        cb_reserve_back(cb_tile, tiles_per_block);
        for (uint32_t r = 0; r < block_ht; ++r) {
            tilize_block(cb_in, block_wt, cb_tile, r * block_wt, r * block_wt);
        }
        cb_push_back(cb_tile, tiles_per_block);
        cb_pop_front(cb_in, tiles_per_block);
        tilize_uninit(cb_in, cb_tile);

        cb_wait_front(cb_tile, tiles_per_block);  // read by index; popped after the divide

        // ----- 2a. abs all tiles_per_block into cb_abs (one acquire; needs dst_full_sync for >4) -----
        // Force the SrcA tile-dim/stride reconfig (is_tile_dim_reconfig_en=true): the default reconfig
        // keeps the prior element stride, so after a bf16 tilize the fp32 cb_tile would be read with a
        // 2-byte stride and copy_tile would misread it (corrupting the amax for bf16 input).
        reconfig_data_format_srca<false, true>(cb_tile);
        pack_reconfig_data_format(cb_abs);
        copy_tile_init(cb_tile);
        abs_tile_init();
        cb_reserve_back(cb_abs, tiles_per_block);
        for (uint32_t c = 0; c < tiles_per_block; c += acq_tiles) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < acq_tiles; ++j) {
                copy_tile(cb_tile, c + j, j);
                abs_tile(j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < acq_tiles; ++j) {
                pack_tile(j, cb_abs);
            }
            tile_regs_release();
        }
        cb_push_back(cb_abs, tiles_per_block);

        // ----- 2b. reduce -> per-row amax -> scale + 1/scale, all rows under one acquire -----
        // Row r reduces its block_wt tiles into dst slot 2*r (FPU MAX-pool accumulates across tiles),
        // then clamp / *1/448 -> scale (slot 2*r); copy to slot 2*r+1 and recip -> 1/scale.
        cb_wait_front(cb_abs, tiles_per_block);
        cb_reserve_back(cb_scale_tiles, block_ht);
        cb_reserve_back(cb_inv_scale_tiles, block_ht);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_scaler, cb_scale_tiles);
        for (uint32_t r = 0; r < block_ht; ++r) {
            for (uint32_t k = 0; k < block_wt; ++k) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_scaler, r * block_wt + k, 0, 2 * r);
            }
        }
        reduce_uninit();
        clamp_tile_init();
        for (uint32_t r = 0; r < block_ht; ++r) {
            clamp_tile(2 * r, clamp_min_bits, clamp_max_bits);  // slot 2*r = clamp(amax)
        }
        binop_with_scalar_tile_init();
        for (uint32_t r = 0; r < block_ht; ++r) {
            mul_unary_tile(2 * r, inv_448_bits);  // slot 2*r = scale = clamp(amax)/448
        }
        copy_dest_values_init();
        for (uint32_t r = 0; r < block_ht; ++r) {
            copy_dest_values<DataFormat::Float32>(2 * r, 2 * r + 1);  // slot 2*r+1 = scale
        }
        recip_tile_init();
        for (uint32_t r = 0; r < block_ht; ++r) {
            recip_tile(2 * r + 1);  // slot 2*r+1 = 1/scale (col 0 valid; other cols unused by bcast)
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < block_ht; ++r) {
            pack_tile(2 * r, cb_scale_tiles);          // scale output (row order)
            pack_tile(2 * r + 1, cb_inv_scale_tiles);  // 1/scale for the divide (row order)
        }
        tile_regs_release();
        cb_push_back(cb_scale_tiles, block_ht);
        cb_push_back(cb_inv_scale_tiles, block_ht);
        cb_pop_front(cb_abs, tiles_per_block);

        // ----- 3+4 fused: divide (cb_tile * bcast_col(1/scale)) then pack_untilize DST -> e4m3 -----
        // mul_tiles_bcast_cols leaves the divided tiles in DST; pack_untilize_dest untilizes them
        // straight into cb_output_e4m3, skipping the cb_out_tile L1 round-trip and the separate
        // untilize math pass. Each acquire holds acq_tiles tiles == acq_tiles/block_wt tile-rows.
        reconfig_data_format(cb_tile, cb_inv_scale_tiles);
        mul_bcast_cols_init_short(cb_tile, cb_inv_scale_tiles);
        pack_untilize_dest_init<block_wt>(cb_output_e4m3);
        cb_wait_front(cb_inv_scale_tiles, block_ht);
        for (uint32_t c = 0; c < tiles_per_block; c += acq_tiles) {
            cb_reserve_back(cb_output_e4m3, acq_tiles);
            tile_regs_acquire();
            for (uint32_t j = 0; j < acq_tiles; ++j) {
                const uint32_t gt = c + j;         // global tile index in the block
                const uint32_t r = gt / block_wt;  // its tile-row -> its 1/scale tile
                mul_tiles_bcast_cols(cb_tile, cb_inv_scale_tiles, gt, r, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<block_wt>(cb_output_e4m3, acq_tiles / block_wt);
            tile_regs_release();
            cb_push_back(cb_output_e4m3, acq_tiles);
        }
        pack_untilize_uninit(cb_output_e4m3);
        cb_pop_front(cb_tile, tiles_per_block);
        cb_pop_front(cb_inv_scale_tiles, block_ht);
    }
}
