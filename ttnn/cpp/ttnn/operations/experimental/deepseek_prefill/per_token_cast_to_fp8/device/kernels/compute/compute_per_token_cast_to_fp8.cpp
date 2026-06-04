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
//   3. divide: out_tile = cb_tile * bcast_col(cb_inv_scale_tiles) per tile -> cb_out_tile.
//   4. untilize cb_out_tile -> cb_output_e4m3 (scaled, cast to e4m3).
// The writer extracts column 0 of cb_scale_tiles into the scale output [..., M, H/128].
//
// fp32_dest_acc_en=True (required for e4m3 on Blackhole; also gives fp32 reduce/divide precision).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"  // abs_tile / abs_tile_init
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tile = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_abs = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t cb_inv_scale_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_tile = get_compile_time_arg_val(6);
    constexpr uint32_t cb_output_e4m3 = get_compile_time_arg_val(7);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(8);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(9);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(10);

    // Tile width from the tensor's tile spec.
    constexpr uint32_t block_w = 128;  // BlockW
    constexpr uint32_t tile_w = get_compile_time_arg_val(11);
    constexpr uint32_t block_wt = block_w / tile_w;  // BlockWt
    constexpr uint32_t block_ht = 1;                 // BlockHt
    constexpr uint32_t tiles_per_block = block_ht * block_wt;

    constexpr uint32_t IDST0 = 0;
    constexpr uint32_t IDST1 = 1;

    uint32_t num_blocks = get_arg_val<uint32_t>(0);  // tile_h x 128 blocks for this core

    // Configure the unpacker hw on the fp32 reduce/abs operand so num_faces / tile dims are full
    // (4 faces). Configuring on a bf16 cb_in instead leaves the fp32 reduce reading only 2 faces
    // (within-tile column reduce sees cols 0-15) for bf16 input; tilize_init re-inits for cb_in.
    compute_kernel_hw_startup(cb_abs, cb_output_e4m3);
    cb_wait_front(cb_scaler, 1);  // reader-filled 1.0 scaler, reused for every reduce

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        {
            // ----- 1. tilize input row-major -> tile -----
            reconfig_data_format_srca(cb_in);
            pack_reconfig_data_format(cb_tile);
            tilize_init(cb_in, tiles_per_block, cb_tile);
            cb_wait_front(cb_in, tiles_per_block);
            cb_reserve_back(cb_tile, tiles_per_block);
            tilize_block(cb_in, tiles_per_block, cb_tile);
            cb_push_back(cb_tile, tiles_per_block);
            cb_pop_front(cb_in, tiles_per_block);
            tilize_uninit(cb_in, cb_tile);

            // ----- 2. block amax -> scale (col 0) and 1/scale (col 0) -----
            cb_wait_front(cb_tile, tiles_per_block);  // read by index; popped after the divide
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                // Abs the block row's tiles into cb_abs. Force the SrcA tile-dim/stride reconfig
                // (is_tile_dim_reconfig_en=true): the default reconfig keeps the prior element
                // stride, so after a bf16 tilize the fp32 cb_tile would be read with a 2-byte
                // stride and copy_tile would misread it (corrupting the amax for bf16 input).
                reconfig_data_format_srca<false, true>(cb_tile);
                pack_reconfig_data_format(cb_abs);
                copy_tile_init(cb_tile);
                cb_reserve_back(cb_abs, block_wt);
                abs_tile_init();
                for (uint32_t k = 0; k < block_wt; ++k) {
                    tile_regs_acquire();
                    copy_tile(cb_tile, block_h_idx * block_wt + k, IDST0);
                    abs_tile(IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_abs);
                    tile_regs_release();
                }
                cb_push_back(cb_abs, block_wt);

                // reduce -> per-row max (col 0), accumulate, clamp, *1/448 -> scale (slot 0);
                // copy to slot 1 and recip -> 1/scale (slot 1). One acquire produces both, so each
                // block row's 1/scale is its own scale (no CB reload, which would always read row 0).
                cb_wait_front(cb_abs, block_wt);
                cb_reserve_back(cb_scale_tiles, 1);
                cb_reserve_back(cb_inv_scale_tiles, 1);
                tile_regs_acquire();
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_scaler, cb_scale_tiles);
                for (uint32_t k = 0; k < block_wt; ++k) {
                    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_scaler, k, 0, k);
                }
                reduce_uninit();

                binary_max_tile_init();
                for (uint32_t i = 1; i < block_wt; i++) {
                    binary_max_tile(IDST0, i, IDST0);  // slot 0 = amax
                }

                clamp_tile_init();
                clamp_tile(IDST0, clamp_min_bits, clamp_max_bits);  // slot 0 = clamp(amax)
                binop_with_scalar_tile_init();
                mul_unary_tile(IDST0, inv_448_bits);  // slot 0 = scale = clamp(amax)/448
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(IDST0, IDST1);  // slot 1 = scale
                recip_tile_init();
                recip_tile(IDST1);  // slot 1 = 1/scale (col 0 valid; other cols = 1/0 = inf, unused by bcast)
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(IDST0, cb_scale_tiles);      // scale output
                pack_tile(IDST1, cb_inv_scale_tiles);  // 1/scale for the divide (same fp32 format)
                tile_regs_release();

                cb_push_back(cb_scale_tiles, 1);
                cb_push_back(cb_inv_scale_tiles, 1);

                cb_pop_front(cb_abs, block_wt);
            }

            // ----- 3. divide: cb_out_tile = cb_tile * bcast_col(1/scale) -----
            reconfig_data_format(cb_tile, cb_inv_scale_tiles);
            pack_reconfig_data_format(cb_out_tile);
            mul_bcast_cols_init_short(cb_tile, cb_inv_scale_tiles);
            cb_wait_front(cb_inv_scale_tiles, block_ht);
            cb_reserve_back(cb_out_tile, tiles_per_block);
            for (uint32_t block_h_idx = 0; block_h_idx < block_ht; ++block_h_idx) {
                for (uint32_t k = 0; k < block_wt; ++k) {
                    tile_regs_acquire();
                    mul_tiles_bcast_cols(cb_tile, cb_inv_scale_tiles, block_h_idx * block_wt + k, block_h_idx, IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_out_tile);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_out_tile, tiles_per_block);
            cb_pop_front(cb_tile, tiles_per_block);
            cb_pop_front(cb_inv_scale_tiles, block_ht);

            // ----- 4. untilize cb_out_tile -> output e4m3 (scaled) -----
            reconfig_data_format_srca(cb_out_tile);
            pack_reconfig_data_format(cb_output_e4m3);
            untilize_init(cb_out_tile);
            cb_wait_front(cb_out_tile, tiles_per_block);
            cb_reserve_back(cb_output_e4m3, tiles_per_block);
            untilize_block(cb_out_tile, tiles_per_block, cb_output_e4m3);
            cb_push_back(cb_output_e4m3, tiles_per_block);
            cb_pop_front(cb_out_tile, tiles_per_block);
            untilize_uninit(cb_out_tile);
        }
    }
}
