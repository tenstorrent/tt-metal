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
#include "tools/profiler/kernel_profiler.hpp"

#ifdef FP8_SCALE
// Extra LLKs used by the fused per-token FP8 quantization (mirrors compute_per_token_cast_to_fp8.cpp).
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
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
//   5: block_ct_dim    - tiles per pack call (largest divisor of full_ct_dim <= 8; forced to 4 on FP8_SCALE)
// FP8_SCALE only:
//   6: cb_scaler_id        - reduce scaler (1.0), reader-filled (c_12)
//   7: cb_abs_id           - abs tiles for one 128-element block (c_20)
//   8: cb_scale_tiles_id   - col0 = per-token scale; compute -> writer (c_21)
//   9: cb_inv_scale_tiles_id - col0 = 1/scale, internal (c_22)
//  10: (unused) was cb_out_tile_id; fused divide -> pack_untilize_dest no longer needs it (c_23)
//  11: clamp_min_bits      - bit_cast(1e-4f)
//  12: clamp_max_bits      - bit_cast(3.0e38f)
//  13: inv_e4m3_max_bits   - bit_cast(1.0f / 448.0f)

void kernel_main() {
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(3);
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(4);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(5);

    constexpr uint32_t full_ct_dim = hidden_size / 32;
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;

#ifdef FP8_SCALE
    constexpr uint32_t cb_scaler_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_abs_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_scale_tiles_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_inv_scale_tiles_id = get_compile_time_arg_val(9);
    // arg 10 (cb_out_tile) is no longer read: divide now packs straight from DST via pack_untilize_dest.
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(11);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(12);
    constexpr uint32_t inv_e4m3_max_bits = get_compile_time_arg_val(13);
    // block_ht: number of 128-element scale blocks processed per outer iteration (mirrors the
    // tile_h / BlockHt loop in compute_per_token_cast_to_fp8.cpp). block_ct_dim stays one scale
    // block wide (4 tiles), so every acquire is <= 4 fp32 tiles and half-sync DEST is kept.
    constexpr uint32_t block_ht = get_compile_time_arg_val(14);

    // block_ct_dim == 4 on this path: one pack-untilize block is exactly one 128-element scale block
    // (4 tiles of 32 cols). REDUCE_ROW over those 4 tiles yields one amax per token (per tile row).
    // fp32_dest_acc_en is enabled by the host on the fp8 path (required for e4m3, gives fp32 reduce).
    compute_kernel_hw_startup(cb_abs_id, cb_untilize_id);
    cb_wait_front(cb_scaler_id, 1);  // reader-filled 1.0 scaler, reused for every reduce
#else
    compute_kernel_hw_startup(cb_in_id, cb_untilize_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_untilize_id);
#endif

    while (true) {
        cb_reserve_back(cb_untilize_id, read_batch_size);

        cb_wait_front(cb_signal_id, 1);
        uint32_t val = read_tile_value(cb_signal_id, 0, 0);
        cb_pop_front(cb_signal_id, 1);
        if (val == ROUTE_INFO_SENTINEL) {
            break;
        }

#ifdef FP8_SCALE
        // Process block_ht consecutive 128-element scale blocks per outer iteration. The reader
        // streams block_ht * block_ct_dim tiles into cb_in as one chunk; each scale block keeps its
        // own per-token amax/scale (mirrors the block_h_idx loop in compute_per_token_cast_to_fp8).
        for (uint32_t block = 0; block < num_blocks; block += block_ht) {
            constexpr uint32_t tiles_per_iter = block_ht * block_ct_dim;
            cb_wait_front(cb_in_id, tiles_per_iter);
            DeviceZoneScopedN("DISPATCH-UNTILIZE-BLOCK");

            {
                DeviceZoneScopedN("compute-part");
                // ----- 1. abs all block_ht scale blocks into cb_abs (block_ct_dim tiles per acquire) -----
                // block_ct_dim == 4 keeps each acquire within the fp32 half-sync DEST budget.
                reconfig_data_format_srca<false, true>(cb_in_id);
                pack_reconfig_data_format(cb_abs_id);
                copy_tile_init(cb_in_id);
                abs_tile_init();
                cb_reserve_back(cb_abs_id, tiles_per_iter);
                for (uint32_t c = 0; c < tiles_per_iter; c += block_ct_dim) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        copy_tile(cb_in_id, c + k, k);
                        abs_tile(k);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        pack_tile(k, cb_abs_id);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_abs_id, tiles_per_iter);

                // ----- 2. per-block amax -> scale + 1/scale, all block_ht blocks under one acquire -----
                // Block h reduces its block_ct_dim tiles into dst slot 2*h (MAX-pool accumulates across
                // tiles), then clamp / *1/448 -> scale (slot 2*h); copy to slot 2*h+1 and recip -> 1/scale.
                // 2*block_ht slots fit the fp32 half-sync DEST budget at block_ht == 2.
                cb_wait_front(cb_abs_id, tiles_per_iter);
                cb_reserve_back(cb_scale_tiles_id, block_ht);
                cb_reserve_back(cb_inv_scale_tiles_id, block_ht);
                tile_regs_acquire();
                reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_scale_tiles_id);
                for (uint32_t h = 0; h < block_ht; ++h) {
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(
                            cb_abs_id, cb_scaler_id, h * block_ct_dim + k, 0, 2 * h);
                    }
                }
                reduce_uninit();
                clamp_tile_init();
                for (uint32_t h = 0; h < block_ht; ++h) {
                    clamp_tile(2 * h, clamp_min_bits, clamp_max_bits);  // slot 2*h = clamp(amax)
                }
                binop_with_scalar_tile_init();
                for (uint32_t h = 0; h < block_ht; ++h) {
                    mul_unary_tile(2 * h, inv_e4m3_max_bits);  // slot 2*h = scale = clamp(amax)/448
                }
                copy_dest_values_init();
                for (uint32_t h = 0; h < block_ht; ++h) {
                    copy_dest_values<DataFormat::Float32>(2 * h, 2 * h + 1);  // slot 2*h+1 = scale
                }
                recip_tile_init();
                for (uint32_t h = 0; h < block_ht; ++h) {
                    recip_tile(2 * h + 1);  // slot 2*h+1 = 1/scale (col 0 valid; other cols unused by bcast)
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t h = 0; h < block_ht; ++h) {
                    pack_tile(2 * h, cb_scale_tiles_id);          // per-token scales (col 0) -> writer
                    pack_tile(2 * h + 1, cb_inv_scale_tiles_id);  // 1/scale for the divide
                }
                tile_regs_release();
                cb_push_back(cb_scale_tiles_id, block_ht);
                cb_push_back(cb_inv_scale_tiles_id, block_ht);
                cb_pop_front(cb_abs_id, tiles_per_iter);

                // ----- 3+4 fused: divide (cb_in * bcast_col(1/scale)) then pack_untilize DST -> e4m3 -----
                // mul_tiles_bcast_cols leaves the divided tiles in DST; pack_untilize_dest untilizes them
                // straight into cb_untilize, skipping the cb_out_tile L1 round-trip and the separate
                // untilize math pass (mirrors compute_per_token_cast_to_fp8.cpp). The pack target column is
                // (block + h) per scale block, so this stays one block per acquire -- unlike the standalone
                // op it cannot batch the pack across blocks, since those map to different hidden columns.
                reconfig_data_format(cb_in_id, cb_inv_scale_tiles_id);
                mul_bcast_cols_init_short(cb_in_id, cb_inv_scale_tiles_id);
                pack_untilize_dest_init<block_ct_dim, full_ct_dim>(cb_untilize_id);
                cb_wait_front(cb_inv_scale_tiles_id, block_ht);
                for (uint32_t h = 0; h < block_ht; ++h) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < block_ct_dim; ++k) {
                        mul_tiles_bcast_cols(cb_in_id, cb_inv_scale_tiles_id, h * block_ct_dim + k, h, k);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_untilize_dest<block_ct_dim, full_ct_dim>(
                        cb_untilize_id, /*block_rt_dim=*/1, /*block_c_index=*/block + h);
                    tile_regs_release();
                }
                pack_untilize_uninit(cb_untilize_id);
                cb_pop_front(cb_in_id, tiles_per_iter);
                cb_pop_front(cb_inv_scale_tiles_id, block_ht);
            }
        }
#else
        for (uint32_t block = 0; block < num_blocks; block++) {
            DeviceZoneScopedN("DISPATCH-UNTILIZE");
            cb_wait_front(cb_in_id, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_untilize_id, block);
            cb_pop_front(cb_in_id, block_ct_dim);
        }
#endif

        cb_push_back(cb_untilize_id, read_batch_size);
    }
#ifndef FP8_SCALE
    pack_untilize_uninit(cb_untilize_id);
#endif
}
