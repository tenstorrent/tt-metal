// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/debug/dprint.h"

#ifdef FP8_PER_TOKEN_SCALE
// Per-token FP8 scale path: amax -> scale/1-over-scale -> divide, grafted onto the untilize compute.
// Ported from compute_per_token_cast_to_fp8.cpp (tilize stage dropped: dispatch input is already tiled).
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

// Compile-time args:
//   0: cb_signal_id    - CB for reader->compute signaling (c_10)
//   1: cb_untilize_id  - CB for compute untilized output (c_11)
//   2: cb_in_id        - CB for untilize input tile data (c_0)
//   3: hidden_size     - hidden dimension (e.g., 7168)
//   4: read_batch_size - number of rows per untilize batch (32)
//   5: block_ct_dim    - tiles per pack call (largest divisor of full_ct_dim <= 8)

void kernel_main() {
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(3);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(4);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(5);

    constexpr uint32_t full_ct_dim = hidden_size / 32;
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;

#ifdef FP8_PER_TOKEN_SCALE
    // ===== Per-token FP8 scale path =====
    // Each 128-element scale block = block_wt(4) tiles; a pack-block (block_ct_dim tiles) holds
    // sb_per_block whole scale blocks (block_ct_dim is forced to a multiple of 4 on the host). For
    // each scale block: |x| -> row-max amax (col0) -> scale=clamp(amax,1e-4)/448 and 1/scale ->
    // divide the block's tiles by bcast(1/scale) into cb_out_tile. The pack-block is then
    // pack_untilized from the divided tiles into the row-major FP8 output. Scales (col0) go to the
    // writer via cb_scale_tiles for appending to the metadata tail.
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(6);
    constexpr uint32_t cb_abs = get_compile_time_arg_val(7);
    constexpr uint32_t cb_scale_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t cb_inv_scale_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out_tile = get_compile_time_arg_val(10);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(11);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(12);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(13);

    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t BLOCK_W = 128;
    constexpr uint32_t block_wt = BLOCK_W / TILE_W;             // 4 tiles per 128-element scale block
    constexpr uint32_t sb_per_block = block_ct_dim / block_wt;  // whole scale blocks per pack-block
    constexpr uint32_t IDST0 = 0;
    constexpr uint32_t IDST1 = 1;

    compute_kernel_hw_startup(cb_abs, cb_untilize_id);
    cb_wait_front(cb_scaler, 1);  // reader-filled 1.0 scaler, reused for every reduce

    while (true) {
        cb_reserve_back(cb_untilize_id, read_batch_size);

        cb_wait_front(cb_signal_id, 1);
        uint32_t val = read_tile_value(cb_signal_id, 0, 0);
        cb_pop_front(cb_signal_id, 1);
        if (val == ROUTE_INFO_SENTINEL) {
            break;
        }

        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_wait_front(cb_in_id, block_ct_dim);
            cb_reserve_back(cb_out_tile, block_ct_dim);

            for (uint32_t sb = 0; sb < sb_per_block; sb++) {
                const uint32_t base = sb * block_wt;  // tile offset of this scale block in the pack-block

                // ----- 1. abs the scale block's tiles -> cb_abs -----
                reconfig_data_format_srca<false, true>(cb_in_id);
                pack_reconfig_data_format(cb_abs);
                copy_tile_init(cb_in_id);
                cb_reserve_back(cb_abs, block_wt);
                abs_tile_init();
                for (uint32_t k = 0; k < block_wt; ++k) {
                    tile_regs_acquire();
                    copy_tile(cb_in_id, base + k, IDST0);
                    abs_tile(IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_abs);
                    tile_regs_release();
                }
                cb_push_back(cb_abs, block_wt);

                // ----- 2. row-max amax (col0) -> scale and 1/scale -----
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
                for (uint32_t i = 1; i < block_wt; ++i) {
                    binary_max_tile(IDST0, i, IDST0);  // slot 0 = amax over the block_wt tiles
                }
                clamp_tile_init();
                clamp_tile(IDST0, clamp_min_bits, clamp_max_bits);  // clamp(amax)
                binop_with_scalar_tile_init();
                mul_unary_tile(IDST0, inv_448_bits);  // slot 0 = scale = clamp(amax)/448
                copy_dest_values_init();
                copy_dest_values<DataFormat::Float32>(IDST0, IDST1);  // slot 1 = scale
                recip_tile_init();
                recip_tile(IDST1);  // slot 1 = 1/scale (col0 valid; bcast uses col0)
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_scale_tiles);      // scale -> writer (metadata tail)
                pack_tile(IDST1, cb_inv_scale_tiles);  // 1/scale -> divide
                tile_regs_release();
                cb_push_back(cb_scale_tiles, 1);
                cb_push_back(cb_inv_scale_tiles, 1);
                cb_pop_front(cb_abs, block_wt);

                // ----- 3. divide: cb_in tiles * bcast_col(1/scale) -> cb_out_tile -----
                reconfig_data_format(cb_in_id, cb_inv_scale_tiles);
                pack_reconfig_data_format(cb_out_tile);
                mul_bcast_cols_init_short(cb_in_id, cb_inv_scale_tiles);
                cb_wait_front(cb_inv_scale_tiles, 1);
                for (uint32_t k = 0; k < block_wt; ++k) {
                    tile_regs_acquire();
                    mul_tiles_bcast_cols(cb_in_id, cb_inv_scale_tiles, base + k, 0, IDST0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(IDST0, cb_out_tile);
                    tile_regs_release();
                }
                cb_pop_front(cb_inv_scale_tiles, 1);
            }
            cb_push_back(cb_out_tile, block_ct_dim);

            // ----- 4. pack_untilize the divided pack-block -> row-major FP8 output -----
            reconfig_data_format_srca(cb_out_tile);
            pack_reconfig_data_format(cb_untilize_id);
            pack_untilize_init<block_ct_dim, full_ct_dim>(cb_out_tile, cb_untilize_id);
            cb_wait_front(cb_out_tile, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_out_tile, 1, cb_untilize_id, block);
            pack_untilize_uninit(cb_untilize_id);
            cb_pop_front(cb_out_tile, block_ct_dim);
            cb_pop_front(cb_in_id, block_ct_dim);
        }

        cb_push_back(cb_untilize_id, read_batch_size);
    }
#else
    compute_kernel_hw_startup(cb_in_id, cb_untilize_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_untilize_id);

    while (true) {
        cb_reserve_back(cb_untilize_id, read_batch_size);

        cb_wait_front(cb_signal_id, 1);
        uint32_t val = read_tile_value(cb_signal_id, 0, 0);
        cb_pop_front(cb_signal_id, 1);
        if (val == ROUTE_INFO_SENTINEL) {
            break;
        }

        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_wait_front(cb_in_id, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_untilize_id, block);
            cb_pop_front(cb_in_id, block_ct_dim);
        }

        cb_push_back(cb_untilize_id, read_batch_size);
    }
    pack_untilize_uninit(cb_untilize_id);
#endif
}
