// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/bcast.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/tile_move_copy.h>
#include <compute_kernel_api/eltwise_binary.h>
#include <compute_kernel_api/reduce.h>

template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
inline void reduce_tile_to_cb(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t size) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    tile_regs_acquire();
    cb_wait_front(icb1, onetile);

    reduce_init<reduce_type, reduce_dim>(ocb, icb0, icb1);
    for (uint32_t x = 0; x < size; ++x) {
        cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        reduce_tile<reduce_type, reduce_dim>(icb0, icb1, x, bcast_scaler0, dst0);
    }
    reduce_uninit();
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    cb_push_back(ocb, onetile);
}

// Multiply two tiles from CBs
// Note: caller must ensure tiles are available via cb_wait_front before calling
ALWI void elementwise_multiply(
    uint32_t src0_cb_id, uint32_t src1_cb_id, uint32_t out_cb_id, uint32_t tile_index_0, uint32_t tile_index_1) {
    constexpr uint32_t dst_reg_tile = 0;

    cb_reserve_back(out_cb_id, /*ntiles*/ 1);

    tile_regs_acquire();
    // Multiply src0_cb_id * src1_cb_id
    mul_tiles(src0_cb_id, src1_cb_id, tile_index_0, tile_index_1, dst_reg_tile);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst_reg_tile, out_cb_id);
    tile_regs_release();

    cb_push_back(out_cb_id, /*ntiles*/ 1);
}

// TODO: first candidate for fusion
// Compute: output = src0 * (src1 - scalar_cb)
// This computes y * (grad - sum(y * grad)) for softmax backward
ALWI void fused_sub_mul(
    uint32_t src0_cb_id,        // y (softmax output)
    uint32_t src1_cb_id,        // grad (upstream gradient)
    uint32_t sum_reduce_cb_id,  // sum(y * grad) - broadcasted scalar
    uint32_t intermed_cb_id,    // intermediate CB for grad - sum
    uint32_t out_cb_id,         // output
    uint32_t src0_tile_idx,
    uint32_t src1_tile_idx) {
    // Step 1: Compute grad - sum(y * grad) and store in intermediate CB
    // Note: caller ensures src1, sum_reduce_cb have required tiles available
    // sum_reduce_cb always has the scalar at index 0
    cb_reserve_back(intermed_cb_id, 1);

    sub_bcast_cols_init_short(src1_cb_id, sum_reduce_cb_id);

    tile_regs_acquire();
    sub_tiles_bcast<BROADCAST_TYPE>(src1_cb_id, sum_reduce_cb_id, src1_tile_idx, 0, 0);  // grad[w] - sum[0]
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, intermed_cb_id);
    tile_regs_release();

    cb_push_back(intermed_cb_id, 1);

    // Step 2: Compute y * (grad - sum)
    // Note: caller ensures src0 has required tiles available
    cb_reserve_back(out_cb_id, 1);
    cb_wait_front(intermed_cb_id, 1);

    mul_tiles_init(src0_cb_id, intermed_cb_id);

    tile_regs_acquire();
    mul_tiles(src0_cb_id, intermed_cb_id, src0_tile_idx, 0, 0);  // y[w] * intermed[0]
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, out_cb_id);
    tile_regs_release();

    cb_pop_front(intermed_cb_id, 1);
    cb_push_back(out_cb_id, 1);
}

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t y_cb_id = get_compile_time_arg_val(0);               // softmax_output (y)
    constexpr uint32_t grad_cb_id = get_compile_time_arg_val(1);            // upstream_grad (grad)
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);             // output
    constexpr uint32_t mul_cb_id = get_compile_time_arg_val(3);             // y * grad
    constexpr uint32_t sum_reduce_cb_id = get_compile_time_arg_val(4);      // sum(y * grad)
    constexpr uint32_t grad_minus_sum_cb_id = get_compile_time_arg_val(5);  // grad - sum(y * grad)
    constexpr uint32_t scaler_cb_id = get_compile_time_arg_val(6);          // scaler for reduction (1.0)
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(7);

    // Runtime args
    const uint32_t num_rows = get_arg_val<uint32_t>(0);        // Number of rows to process
    const uint32_t width_in_tiles = get_arg_val<uint32_t>(1);  // Tiles per row

    // Initialize compute operations
    binary_op_init_common(y_cb_id, grad_cb_id, out_cb_id);

    // Process each row
    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for reader to provide all input tiles
        cb_wait_front(y_cb_id, width_in_tiles);
        cb_wait_front(grad_cb_id, width_in_tiles);

        // Step 1: Compute y * grad for all tiles in the row (element-wise multiplication)
        mul_tiles_init(y_cb_id, grad_cb_id);
        for (uint32_t w = 0; w < width_in_tiles; ++w) {
            elementwise_multiply(y_cb_id, grad_cb_id, mul_cb_id, w, w);
        }

        // Step 2: Reduce sum(y * grad) across the row using scaler CB
        reduce_tile_to_cb(mul_cb_id, scaler_cb_id, sum_reduce_cb_id, width_in_tiles);
        cb_pop_front(mul_cb_id, width_in_tiles);

        // Step 3: For each tile in the row, compute final result: y * (grad - sum(y * grad))
        // Wait for all input tiles to be available before processing
        cb_wait_front(y_cb_id, width_in_tiles);
        cb_wait_front(grad_cb_id, width_in_tiles);
        cb_wait_front(sum_reduce_cb_id, 1);

        // Step 3: subtract grad - sum, then multiply y * (grad - sum)
        for (uint32_t w = 0; w < width_in_tiles; ++w) {
            fused_sub_mul(
                y_cb_id,               // y
                grad_cb_id,            // grad
                sum_reduce_cb_id,      // sum(y * grad)
                grad_minus_sum_cb_id,  // intermediate: grad - sum
                out_cb_id,             // output
                w,                     // y tile index
                w);                    // grad tile index
        }

        // Pop consumed data for this row
        cb_pop_front(y_cb_id, width_in_tiles);
        cb_pop_front(grad_cb_id, width_in_tiles);
        cb_pop_front(sum_reduce_cb_id, 1);
    }
}
}  // namespace NAMESPACE
