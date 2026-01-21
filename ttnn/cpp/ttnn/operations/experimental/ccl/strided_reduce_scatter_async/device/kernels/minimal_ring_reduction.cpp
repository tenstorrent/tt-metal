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

    // Let's set some particular values for the params used
    const uint32_t M_blocks_per_core = 1;
    const uint32_t chunk_counts_per_width = 1;
    const uint32_t mm_N_blocks_per_slice = 1;
    const uint32_t batch_size = input_tensor_B;
    const uint32_t chunks_per_mm_N_block = 1;
    const uint32_t chunk_width = 2;

    // Initialize binary operations - use the same constants consistently
    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            for (uint32_t chunk_idx = 0; chunk_idx < chunks_per_mm_N_block; chunk_idx++) {
                for (uint32_t i = 1; i < ring_size; i++) {
                    for (uint32_t chunk_piece_idx = 0; chunk_piece_idx < mm_N_blocks_per_slice; chunk_piece_idx++) {
                        uint32_t tiles_to_read_in_current_direction = chunk_width;
                        cb_wait_front(input_cb_id, tile_granularity);
                        cb_wait_front(intermediate_cb, tile_granularity);
                        // cb_reserve_back(output_cb, tile_granularity);
                        acquire_dst();
                        for (uint32_t tile_id = 0; tile_id < tiles_to_read_in_current_direction; tile_id++) {
                            add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                            // pack_tile(tile_id, output_cb);
                        }
                        release_dst();
                        cb_pop_front(input_cb_id, tile_granularity);
                        cb_pop_front(intermediate_cb, tile_granularity);
                        // cb_push_back(output_cb, tile_granularity);
                    }
                }
            }
        }
    }
}
}  // namespace NAMESPACE
