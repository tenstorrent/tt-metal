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
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
    CircularBuffer cb_in(cb_in_id);
    constexpr uint32_t cb_tile_id = get_compile_time_arg_val(1);
    CircularBuffer cb_tile(cb_tile_id);
    constexpr uint32_t cb_scaler_id = get_compile_time_arg_val(2);
    CircularBuffer cb_scaler(cb_scaler_id);
    constexpr uint32_t cb_abs_id = get_compile_time_arg_val(3);
    CircularBuffer cb_abs(cb_abs_id);
    constexpr uint32_t cb_scale_tiles_id = get_compile_time_arg_val(4);
    CircularBuffer cb_scale_tiles(cb_scale_tiles_id);
    constexpr uint32_t cb_inv_scale_tiles_id = get_compile_time_arg_val(5);
    CircularBuffer cb_inv_scale_tiles(cb_inv_scale_tiles_id);
    constexpr uint32_t cb_output_e4m3_id = get_compile_time_arg_val(6);
    CircularBuffer cb_output_e4m3(cb_output_e4m3_id);
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
    compute_kernel_hw_startup(cb_abs_id, cb_output_e4m3_id);
    cb_scaler.wait_front(1);  // reader-filled 1.0 scaler, reused for every reduce

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        // ----- 1. tilize input row-major -> tile (one tilize_block per tile-row) -----
        // The reader fills cb_in as [block_ht*tile_h x 128] row-major; tilize_block treats each
        // block_wt-tile run as one [tile_h x 128] tile-row, so tile-row r reads/writes tiles
        // [r*block_wt, r*block_wt+block_wt).
        reconfig_data_format_srca(cb_in_id);
        pack_reconfig_data_format(cb_tile_id);
        tilize_init(cb_in_id, block_wt, cb_tile_id);
        cb_in.wait_front(tiles_per_block);
        cb_tile.reserve_back(tiles_per_block);
        for (uint32_t r = 0; r < block_ht; ++r) {
            tilize_block(cb_in_id, block_wt, cb_tile_id, r * block_wt, r * block_wt);
        }
        cb_tile.push_back(tiles_per_block);
        cb_in.pop_front(tiles_per_block);
        tilize_uninit(cb_in_id, cb_tile_id);

        cb_tile.wait_front(tiles_per_block);  // read by index; popped after the divide

        // ----- 2a. abs all tiles_per_block into cb_abs (one acquire; needs dst_full_sync for >4) -----
        // Force the SrcA tile-dim/stride reconfig (is_tile_dim_reconfig_en=true): the default reconfig
        // keeps the prior element stride, so after a bf16 tilize the fp32 cb_tile would be read with a
        // 2-byte stride and copy_tile would misread it (corrupting the amax for bf16 input).
        reconfig_data_format_srca<false, true>(cb_tile_id);
        pack_reconfig_data_format(cb_abs_id);
        copy_tile_init(cb_tile_id);
        abs_tile_init();
        cb_abs.reserve_back(tiles_per_block);
        for (uint32_t c = 0; c < tiles_per_block; c += acq_tiles) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < acq_tiles; ++j) {
                copy_tile(cb_tile_id, c + j, j);
                abs_tile(j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < acq_tiles; ++j) {
                pack_tile(j, cb_abs_id);
            }
            tile_regs_release();
        }
        cb_abs.push_back(tiles_per_block);

        // ----- 2b. reduce -> per-row amax -> scale + 1/scale, all rows under one acquire -----
        // Row r reduces its block_wt tiles into dst slot 2*r (FPU MAX-pool accumulates across tiles),
        // then clamp / *1/448 -> scale (slot 2*r); copy to slot 2*r+1 and recip -> 1/scale.
        cb_abs.wait_front(tiles_per_block);
        cb_scale_tiles.reserve_back(block_ht);
        cb_inv_scale_tiles.reserve_back(block_ht);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_scale_tiles_id);
        for (uint32_t r = 0; r < block_ht; ++r) {
            for (uint32_t k = 0; k < block_wt; ++k) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, r * block_wt + k, 0, 2 * r);
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
            pack_tile(2 * r, cb_scale_tiles_id);          // scale output (row order)
            pack_tile(2 * r + 1, cb_inv_scale_tiles_id);  // 1/scale for the divide (row order)
        }
        tile_regs_release();
        cb_scale_tiles.push_back(block_ht);
        cb_inv_scale_tiles.push_back(block_ht);
        cb_abs.pop_front(tiles_per_block);

        // ----- 3+4 fused: divide (cb_tile * bcast_col(1/scale)) then pack_untilize DST -> e4m3 -----
        // mul_tiles_bcast_cols leaves the divided tiles in DST; pack_untilize_dest untilizes them
        // straight into cb_output_e4m3, skipping the cb_out_tile L1 round-trip and the separate
        // untilize math pass. Each acquire holds acq_tiles tiles == acq_tiles/block_wt tile-rows.
        reconfig_data_format(cb_tile_id, cb_inv_scale_tiles_id);
        mul_bcast_cols_init_short(cb_tile_id, cb_inv_scale_tiles_id);
        pack_untilize_dest_init<block_wt>(cb_output_e4m3_id);
        cb_inv_scale_tiles.wait_front(block_ht);
        for (uint32_t c = 0; c < tiles_per_block; c += acq_tiles) {
            cb_output_e4m3.reserve_back(acq_tiles);
            tile_regs_acquire();
            for (uint32_t j = 0; j < acq_tiles; ++j) {
                const uint32_t gt = c + j;         // global tile index in the block
                const uint32_t r = gt / block_wt;  // its tile-row -> its 1/scale tile
                mul_tiles_bcast_cols(cb_tile_id, cb_inv_scale_tiles_id, gt, r, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_untilize_dest<block_wt>(cb_output_e4m3_id, acq_tiles / block_wt);
            tile_regs_release();
            cb_output_e4m3.push_back(acq_tiles);
        }
        pack_untilize_uninit(cb_output_e4m3_id);
        cb_tile.pop_front(tiles_per_block);
        cb_inv_scale_tiles.pop_front(block_ht);
    }
}
