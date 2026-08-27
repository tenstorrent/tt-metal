// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// REDUCE_OP, REDUCE_DIM and EXP_APPROX_MODE come from the program factory as kernel
// defines -- compute_common.hpp reads them as macros.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "compute_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/neighborhood_kernel_args.hpp"

// Flash attention over one query brick at a time.
//
// READABILITY INVARIANT: this file contains no neighborhood concepts. No context window, no
// stride, no brick geometry, no volume. It receives a tile row of queries, a stream of key
// and value tiles, and an additive mask, and runs online softmax over them. Everything that
// decides WHICH tiles arrive lives in the reader.
//
// The shape is simpler than the general SDPA kernel because a query chunk here is always
// exactly one tile row -- one brick is 32 sites is one tile. So there is no query
// sub-blocking, and the running statistics are one tile each.

namespace kernel_args = ttnn::transformer::neighborhood::kernel_args;

void kernel_main() {
    constexpr uint32_t head_dim_tiles = get_compile_time_arg_val(kernel_args::compute_arg::head_dim_tiles);
    constexpr uint32_t tiles_per_kv_chunk = get_compile_time_arg_val(kernel_args::compute_arg::tiles_per_kv_chunk);
    constexpr uint32_t kv_chunk_count = get_compile_time_arg_val(kernel_args::compute_arg::kv_chunk_count);
    constexpr uint32_t scale_as_float_bits = get_compile_time_arg_val(kernel_args::compute_arg::scale_as_float_bits);
    constexpr uint32_t scores_subblock_width =
        get_compile_time_arg_val(kernel_args::compute_arg::scores_subblock_width);
    constexpr uint32_t scores_subblock_count =
        get_compile_time_arg_val(kernel_args::compute_arg::scores_subblock_count);
    constexpr uint32_t output_subblock_width =
        get_compile_time_arg_val(kernel_args::compute_arg::output_subblock_width);
    constexpr uint32_t output_subblock_count =
        get_compile_time_arg_val(kernel_args::compute_arg::output_subblock_count);

    const uint32_t work_item_count = get_arg_val<uint32_t>(0);

    // One brick of queries is one tile row, and one query CHUNK is this many bricks. Every row
    // in the chunk shares a single query group, hence a single window -- which is why the mask
    // is one tile per gather slot and broadcasts down the rows rather than being stored per row.
    constexpr uint32_t query_tile_rows = get_compile_time_arg_val(kernel_args::compute_arg::query_tile_rows);
    // 0 = the broadcast above. tiles_per_kv_chunk = one mask per brick, which a chunk wider than
    // the stride requires: its bricks each centre a different window.
    constexpr uint32_t mask_subblock_stride = get_compile_time_arg_val(kernel_args::compute_arg::mask_subblock_stride);
    constexpr uint32_t mask_tiles_per_kv_chunk =
        mask_subblock_stride == 0 ? tiles_per_kv_chunk : query_tile_rows * tiles_per_kv_chunk;

    CircularBuffer cb_query(kernel_args::cb_query);
    CircularBuffer cb_key(kernel_args::cb_key);
    CircularBuffer cb_value(kernel_args::cb_value);
    CircularBuffer cb_mask(kernel_args::cb_mask);
    CircularBuffer cb_scores(kernel_args::cb_scores);
    CircularBuffer cb_output(kernel_args::cb_output);
    CircularBuffer cb_reduce_scalar(kernel_args::cb_reduce_scalar);

    compute_kernel_hw_startup<SrcOrder::Reverse>(kernel_args::cb_query, kernel_args::cb_key, kernel_args::cb_output);
    matmul_init(kernel_args::cb_query, kernel_args::cb_key);

    // The reduce identity is produced once by the writer and stays resident.
    cb_reduce_scalar.wait_front(1);

    for (uint32_t work_item = 0; work_item < work_item_count; ++work_item) {
        cb_query.wait_front(query_tile_rows * head_dim_tiles);

        // Ping-pong buffers for the running statistics. Swapped rather than copied each chunk.
        uint32_t current_max = kernel_args::cb_row_max_current;
        uint32_t previous_max = kernel_args::cb_row_max_previous;
        uint32_t current_sum = kernel_args::cb_row_sum_current;
        uint32_t previous_sum = kernel_args::cb_row_sum_previous;
        uint32_t current_output = kernel_args::cb_output_accumulator_current;
        uint32_t previous_output = kernel_args::cb_output_accumulator_previous;

        for (uint32_t kv_chunk_index = 0; kv_chunk_index < kv_chunk_count; ++kv_chunk_index) {
            cb_key.wait_front(tiles_per_kv_chunk * head_dim_tiles);
            cb_mask.wait_front(mask_tiles_per_kv_chunk);

            // scores = Q . K^T, with the additive mask folded in. K arrives as
            // [tiles_per_kv_chunk rows, head_dim_tiles cols], hence transpose.
            reconfig_data_format(kernel_args::cb_query, kernel_args::cb_key);
            pack_reconfig_data_format(kernel_args::cb_scores);
            matmul_blocks(
                kernel_args::cb_query,
                kernel_args::cb_key,
                kernel_args::cb_scores,
                /*M=*/query_tile_rows,
                /*N=*/tiles_per_kv_chunk,
                /*K=*/head_dim_tiles,
                /*num_blocks=*/1,
                /*in0_num_subblocks=*/query_tile_rows,
                /*in1_num_subblocks=*/scores_subblock_count,
                /*in0_block_w=*/head_dim_tiles,
                /*subblock_h=*/1,
                /*subblock_w=*/scores_subblock_width,
                /*transpose=*/true,
                /*add_mask=*/true,
                /*mask_cb=*/kernel_args::cb_mask,
                /*zero_cb=*/kernel_args::cb_zero,
                /*mask_subblock_stride=*/mask_subblock_stride);
            cb_mask.pop_front(mask_tiles_per_kv_chunk);

            // running_max = max(running_max, row_max(scores))
            reduce_c<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                kernel_args::cb_scores,
                kernel_args::cb_reduce_scalar,
                query_tile_rows>(current_max, previous_max, tiles_per_kv_chunk, kv_chunk_index > 0);

            // In place: scores = exp((scores - running_max) * scale), partially reduced into
            // current_sum. The final within-tile row reduction is deferred out of the loop.
            sub_exp_block_bcast_cols_inplace<kernel_args::cb_scores, query_tile_rows, scale_as_float_bits, true>(
                current_max, current_sum, tiles_per_kv_chunk);

            // output = scores . V
            cb_value.wait_front(tiles_per_kv_chunk * head_dim_tiles);
            reconfig_data_format(kernel_args::cb_value, kernel_args::cb_scores);
            pack_reconfig_data_format(current_output);
            matmul_blocks(
                kernel_args::cb_scores,
                kernel_args::cb_value,
                current_output,
                /*M=*/query_tile_rows,
                /*N=*/head_dim_tiles,
                /*K=*/tiles_per_kv_chunk,
                /*num_blocks=*/1,
                /*in0_num_subblocks=*/query_tile_rows,
                /*in1_num_subblocks=*/output_subblock_count,
                /*in0_block_w=*/tiles_per_kv_chunk,
                /*subblock_h=*/1,
                /*subblock_w=*/output_subblock_width,
                /*transpose=*/false);

            // matmul_blocks pops in1 but leaves in0 produced, so the scores have to be
            // retired here or the next chunk deadlocks reserving them.
            cb_scores.pop_front(query_tile_rows * tiles_per_kv_chunk);

            if (kv_chunk_index > 0) {
                // The running max moved, so everything accumulated under the old max has to be
                // rescaled by exp(previous_max - current_max) before it can be added.
                sub_exp_block<scale_as_float_bits>(
                    previous_max, current_max, kernel_args::cb_exp_max_difference, query_tile_rows);
                CircularBuffer(previous_max).pop_front(query_tile_rows);

                mul_tiles_bcast_cols_inplace(previous_sum, kernel_args::cb_exp_max_difference, query_tile_rows);
                add_block_inplace(current_sum, previous_sum, query_tile_rows);

                mul_block_bcast_cols<query_tile_rows, head_dim_tiles, false, true>(
                    previous_output, kernel_args::cb_exp_max_difference, current_output);
            }

            std::swap(previous_max, current_max);
            std::swap(previous_sum, current_sum);
            std::swap(previous_output, current_output);
        }

        // sub_exp only partially reduced each chunk's sum, so finish the within-tile row
        // reduction now -- as a matmul against a column of ones, in place. Reading cb_scores
        // here instead would deadlock: it is retired every chunk.
        matmul_reduce<query_tile_rows>(kernel_args::cb_column_identity, previous_sum);
        recip_block_inplace(previous_sum, query_tile_rows);
        mul_block_bcast_cols<query_tile_rows, head_dim_tiles, true, false>(
            previous_output, previous_sum, kernel_args::cb_output);

        // The running max is the one statistic nothing else retires: the sum is popped by the
        // normalize above, and the accumulators by the rescale. Leaking one tile per work item
        // jams cb_row_max on the third item -- invisible while every core had exactly one,
        // which is every case the correctness test covers.
        CircularBuffer(previous_max).pop_front(query_tile_rows);

        cb_query.pop_front(query_tile_rows * head_dim_tiles);
    }
}
