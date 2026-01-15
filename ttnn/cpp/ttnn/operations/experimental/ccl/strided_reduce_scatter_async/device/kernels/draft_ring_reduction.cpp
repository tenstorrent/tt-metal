// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(5);
    constexpr uint32_t slice_C = get_compile_time_arg_val(6);

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    // Initialize binary operations - use the same constants consistently
    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    uint32_t batch_size = input_tensor_B;
    uint32_t M_blocks_per_core = div_up(M_tiles_per_core, mm_block_ht);

    for (uint32_t b = 0; b < batch_size; b++) {
        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            // each block has a height of mm_block_ht (tiles of matmul block)
            // this is what has to be sent in one step in a single chunk
            for (uint32_t strided_chunk_idx = 0; strided_chunk_idx < chunk_counts_per_slice; strided_chunk_idx++) {
                uint32_t actual_chunk_w = device_chunk_width;
                uint32_t actual_chunk_h = next_mm_aligned_chunk_height(
                    input_chunk_start_tile, M_tiles_per_core, input_tensor_Wt, mm_block_ht);
                uint32_t tiles_in_current_chunk = actual_chunk_w * actual_chunk_h * mm_cores_y;
                // TODO: compute tiles_in_current_chunk for this direction
                // (forward takes the first half, backward takes the second half)
                // adjust input_worker_tile_offset in the backward case accordingly
                // TODO: take into account the workers
                uint32_t num_pages_to_read;  // obtain based on tiles_in_current_chunk and tile_granularity
                // recall the first iteration does not reduce (only sends)
                for (uint32_t i = 0; i < ring_size - 1; i++) {
                    cb_wait_front(input_cb_id, tile_granularity);
                    cb_wait_front(intermediate_cb, tile_granularity);
                    cb_reserve_back(output_cb, tile_granularity);
                    acquire_dst();
                    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                        add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                        pack_tile(tile_id, output_cb);
                    }
                    release_dst();
                    cb_pop_front(input_cb_id, tile_granularity);
                    cb_pop_front(intermediate_cb, tile_granularity);
                    cb_push_back(output_cb, tile_granularity);
                }
            }
        }
    }
}
}  // namespace NAMESPACE
