// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/bcast.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/tile_move_copy.h>
#include <compute_kernel_api/eltwise_binary.h>
#include <compute_kernel_api/eltwise_unary/fill.h>
#include <compute_kernel_api/matmul.h>
#include <compute_kernel_api/mask.h>

// Fused subtract and multiply: output = y * (grad - sum)
// Reuses DST register to eliminate intermediate CB write/read
ALWI void fused_sub_mul(
    uint32_t y_cb_id,           // y (softmax output)
    uint32_t grad_cb_id,        // grad (upstream gradient)
    uint32_t sum_reduce_cb_id,  // sum(y * grad) - broadcasted scalar
    uint32_t out_cb_id,         // output
    uint32_t y_tile_idx,
    uint32_t grad_tile_idx) {
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t dst_reg_tile = 0;

    cb_reserve_back(out_cb_id, one_tile);
    tile_regs_acquire();

    // Step 1: Compute grad - sum(y * grad) and store in DST[0]
    sub_bcast_cols_init_short(grad_cb_id, sum_reduce_cb_id);
    sub_tiles_bcast<BROADCAST_TYPE>(grad_cb_id, sum_reduce_cb_id, grad_tile_idx, 0, dst_reg_tile);

    // Step 2: Multiply y * DST[0], reusing the DST register
    binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(y_cb_id);
    binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(y_cb_id, y_tile_idx, dst_reg_tile);

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst_reg_tile, out_cb_id);
    tile_regs_release();
    cb_push_back(out_cb_id, one_tile);
}

namespace NAMESPACE {
void MAIN {
    // Compile time args
    constexpr uint32_t y_cb_id = get_compile_time_arg_val(0);            // softmax_output (y)
    constexpr uint32_t grad_cb_id = get_compile_time_arg_val(1);         // upstream_grad (grad)
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);          // output
    constexpr uint32_t mul_cb_id = get_compile_time_arg_val(3);          // y * grad
    constexpr uint32_t sum_reduce_cb_id = get_compile_time_arg_val(4);   // sum(y * grad) - accumulated
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(5);         // ones vector for matmul reduction
    constexpr uint32_t batch_sum_cb_id = get_compile_time_arg_val(6);    // batch sum temporary
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(7);  // width in tiles
    constexpr uint32_t mask_w = get_compile_time_arg_val(8);             // padding mask position (0 = no padding)

    // Runtime args
    const uint32_t num_rows = get_arg_val<uint32_t>(0);        // Number of rows to process
    const uint32_t width_in_tiles = get_arg_val<uint32_t>(1);  // Tiles per row

    // Initialize compute operations
    binary_op_init_common(y_cb_id, grad_cb_id, out_cb_id);

    // Two-pass streaming algorithm for minimal L1 memory
    for (uint32_t row = 0; row < num_rows; ++row) {
        // === PASS 1: Streaming reduction to compute sum(y * grad) ===
        constexpr uint32_t dst_product = 0;
        constexpr uint32_t dst_accum = 1;
        constexpr uint32_t one_tile = 1;

        // Adjustable batch size - must match reader kernel
        constexpr uint32_t tiles_per_batch = 4;

        cb_wait_front(ones_cb_id, one_tile);
        cb_reserve_back(sum_reduce_cb_id, one_tile);

        // Create a zero tile
        fill_tile_init();
        tile_regs_acquire();
        fill_tile(dst_accum, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_accum, sum_reduce_cb_id);
        tile_regs_release();
        cb_push_back(sum_reduce_cb_id, one_tile);

        // Process in batches: compute products, then accumulate each batch
        for (uint32_t batch_start = 0; batch_start < width_in_tiles; batch_start += tiles_per_batch) {
            const uint32_t current_batch_size =
                (batch_start + tiles_per_batch <= width_in_tiles) ? tiles_per_batch : (width_in_tiles - batch_start);

            // Wait for this batch from reader
            cb_wait_front(y_cb_id, current_batch_size);
            cb_wait_front(grad_cb_id, current_batch_size);

            // Step 1a: Compute y * grad for all tiles in this batch (elementwise multiplication)
            mul_tiles_init(y_cb_id, grad_cb_id);

            for (uint32_t i = 0; i < current_batch_size; ++i) {
                const uint32_t global_tile_idx = batch_start + i;
                const bool is_last_tile = (global_tile_idx == width_in_tiles - 1);

                cb_reserve_back(mul_cb_id, one_tile);
                tile_regs_acquire();
                mul_tiles(y_cb_id, grad_cb_id, i, i, dst_product);

                // Mask padding in the last tile of the row
                if constexpr (mask_w > 0) {
                    if (is_last_tile) {
                        mask_tile_init();
                        mask_tile(dst_product, mask_w);
                    }
                }

                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst_product, mul_cb_id);
                tile_regs_release();
                cb_push_back(mul_cb_id, one_tile);
            }

            // Step 1b: Reduce this batch to a single sum tile using matmul with ones
            // Write batch sum to temporary CB
            mm_init(mul_cb_id, ones_cb_id, batch_sum_cb_id, /*transpose*/ 0);

            cb_reserve_back(batch_sum_cb_id, one_tile);
            tile_regs_acquire();
            for (uint32_t i = 0; i < current_batch_size; ++i) {
                cb_wait_front(mul_cb_id, i + 1);  // Ensure tile is ready
                matmul_tiles(mul_cb_id, ones_cb_id, i, 0, dst_accum, false);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst_accum, batch_sum_cb_id);
            tile_regs_release();
            cb_push_back(batch_sum_cb_id, one_tile);

            // Step 1c: Add batch sum to running total
            // accumulated_sum = accumulated_sum + batch_sum
            cb_wait_front(sum_reduce_cb_id, one_tile);  // Current accumulated sum
            cb_wait_front(batch_sum_cb_id, one_tile);   // Batch sum we just computed

            add_tiles_init(sum_reduce_cb_id, batch_sum_cb_id);

            tile_regs_acquire();
            add_tiles(sum_reduce_cb_id, batch_sum_cb_id, 0, 0, dst_product);
            tile_regs_commit();
            tile_regs_wait();

            cb_pop_front(sum_reduce_cb_id, one_tile);  // Pop old accumulated sum (CB now empty)
            cb_pop_front(batch_sum_cb_id, one_tile);   // Pop batch sum

            cb_reserve_back(sum_reduce_cb_id, one_tile);  // Reserve slot for new sum
            pack_tile(dst_product, sum_reduce_cb_id);     // Write updated accumulated sum
            tile_regs_release();
            cb_push_back(sum_reduce_cb_id, one_tile);

            // Pop this batch
            cb_pop_front(mul_cb_id, current_batch_size);
            cb_pop_front(y_cb_id, current_batch_size);
            cb_pop_front(grad_cb_id, current_batch_size);
        }

        // === PASS 2: Compute final output with fresh data ===
        cb_wait_front(sum_reduce_cb_id, one_tile);

        for (uint32_t batch_start = 0; batch_start < width_in_tiles; batch_start += tiles_per_batch) {
            const uint32_t current_batch_size =
                (batch_start + tiles_per_batch <= width_in_tiles) ? tiles_per_batch : (width_in_tiles - batch_start);

            // Wait for fresh batch from reader (pass 2 read)
            cb_wait_front(y_cb_id, current_batch_size);
            cb_wait_front(grad_cb_id, current_batch_size);

            // Process each tile: compute y * (grad - sum)
            for (uint32_t i = 0; i < current_batch_size; ++i) {
                fused_sub_mul(
                    y_cb_id,           // y
                    grad_cb_id,        // grad
                    sum_reduce_cb_id,  // sum(y * grad)
                    out_cb_id,         // output
                    i,                 // y tile index (relative to batch)
                    i);                // grad tile index (relative to batch)
            }

            // Pop this batch
            cb_pop_front(y_cb_id, current_batch_size);
            cb_pop_front(grad_cb_id, current_batch_size);
        }

        // Pop sum for this row
        cb_pop_front(sum_reduce_cb_id, one_tile);
    }
}
}  // namespace NAMESPACE
