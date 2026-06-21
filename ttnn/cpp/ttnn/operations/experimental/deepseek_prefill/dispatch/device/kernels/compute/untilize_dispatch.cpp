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

#ifdef FP8_SCALE
// Extra LLKs used by the fused per-token FP8 quantization (mirrors compute_per_token_cast_to_fp8.cpp).
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
//   5: block_ct_dim    - tiles per pack call (largest divisor of full_ct_dim <= 8; forced to 4 on FP8_SCALE)
// FP8_SCALE only:
//   6: cb_scaler_id        - reduce scaler (1.0), reader-filled (c_12)
//   7: cb_abs_id           - abs tiles for one 128-element block (c_20)
//   8: cb_scale_tiles_id   - col0 = per-token scale; compute -> writer (c_21)
//   9: cb_inv_scale_tiles_id - col0 = 1/scale, internal (c_22)
//  10: cb_out_tile_id      - divided tiles -> pack_untilize (c_23)
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
    constexpr uint32_t cb_out_tile_id = get_compile_time_arg_val(10);
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(11);
    constexpr uint32_t clamp_max_bits = get_compile_time_arg_val(12);
    constexpr uint32_t inv_e4m3_max_bits = get_compile_time_arg_val(13);

    constexpr uint32_t IDST0 = 0;
    constexpr uint32_t IDST1 = 1;

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

        for (uint32_t block = 0; block < num_blocks; block++) {
#ifdef FP8_SCALE
            // ----- 1. abs the block's tiles (one 128-element scale block) into cb_abs -----
            reconfig_data_format_srca<false, true>(cb_in_id);
            pack_reconfig_data_format(cb_abs_id);
            copy_tile_init(cb_in_id);
            cb_wait_front(cb_in_id, block_ct_dim);
            cb_reserve_back(cb_abs_id, block_ct_dim);
            abs_tile_init();
            for (uint32_t k = 0; k < block_ct_dim; ++k) {
                tile_regs_acquire();
                copy_tile(cb_in_id, k, IDST0);
                abs_tile(IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_abs_id);
                tile_regs_release();
            }
            cb_push_back(cb_abs_id, block_ct_dim);

            // ----- 2. REDUCE_ROW max -> per-token amax -> clamp -> *1/448 -> scale; recip -> 1/scale -----
            cb_wait_front(cb_abs_id, block_ct_dim);
            cb_reserve_back(cb_scale_tiles_id, 1);
            cb_reserve_back(cb_inv_scale_tiles_id, 1);
            tile_regs_acquire();
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, cb_scale_tiles_id);
            for (uint32_t k = 0; k < block_ct_dim; ++k) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs_id, cb_scaler_id, k, 0, k);
            }
            reduce_uninit();
            binary_max_tile_init();
            for (uint32_t i = 1; i < block_ct_dim; ++i) {
                binary_max_tile(IDST0, i, IDST0);  // slot 0 = amax over the 128-element block
            }
            clamp_tile_init();
            clamp_tile(IDST0, clamp_min_bits, clamp_max_bits);  // slot 0 = clamp(amax)
            binop_with_scalar_tile_init();
            mul_unary_tile(IDST0, inv_e4m3_max_bits);  // slot 0 = scale = clamp(amax)/448
            copy_dest_values_init();
            copy_dest_values<DataFormat::Float32>(IDST0, IDST1);  // slot 1 = scale
            recip_tile_init();
            recip_tile(IDST1);  // slot 1 = 1/scale (col 0 valid; other cols unused by bcast)
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(IDST0, cb_scale_tiles_id);      // per-token scales (col 0) -> writer
            pack_tile(IDST1, cb_inv_scale_tiles_id);  // 1/scale for the divide
            tile_regs_release();
            cb_push_back(cb_scale_tiles_id, 1);
            cb_push_back(cb_inv_scale_tiles_id, 1);
            cb_pop_front(cb_abs_id, block_ct_dim);

            // ----- 3. divide: cb_out_tile = cb_in * bcast_col(1/scale) -----
            reconfig_data_format(cb_in_id, cb_inv_scale_tiles_id);
            pack_reconfig_data_format(cb_out_tile_id);
            mul_bcast_cols_init_short(cb_in_id, cb_inv_scale_tiles_id);
            cb_wait_front(cb_inv_scale_tiles_id, 1);
            cb_reserve_back(cb_out_tile_id, block_ct_dim);
            for (uint32_t k = 0; k < block_ct_dim; ++k) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_in_id, cb_inv_scale_tiles_id, k, 0, IDST0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(IDST0, cb_out_tile_id);
                tile_regs_release();
            }
            cb_push_back(cb_out_tile_id, block_ct_dim);
            cb_pop_front(cb_in_id, block_ct_dim);
            cb_pop_front(cb_inv_scale_tiles_id, 1);

            // ----- 4. pack_untilize the divided block -> e4m3 row-major output at column `block` -----
            reconfig_data_format_srca(cb_out_tile_id);
            pack_reconfig_data_format(cb_untilize_id);
            pack_untilize_init<block_ct_dim, full_ct_dim>(cb_out_tile_id, cb_untilize_id);
            cb_wait_front(cb_out_tile_id, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_out_tile_id, 1, cb_untilize_id, block);
            pack_untilize_uninit(cb_untilize_id);
            cb_pop_front(cb_out_tile_id, block_ct_dim);
#else
            cb_wait_front(cb_in_id, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_untilize_id, block);
            cb_pop_front(cb_in_id, block_ct_dim);
#endif
        }

        cb_push_back(cb_untilize_id, read_batch_size);
    }
#ifndef FP8_SCALE
    pack_untilize_uninit(cb_untilize_id);
#endif
}
