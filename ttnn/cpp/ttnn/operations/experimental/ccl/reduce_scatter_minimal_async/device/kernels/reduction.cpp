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
    constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(3);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);
    constexpr uint32_t num_batches = get_compile_time_arg_val(6);
    constexpr uint32_t num_links = get_compile_time_arg_val(7);

    // const uint32_t num_packets = batch_slice_num_pages / tile_granularity / num_links;
    constexpr uint32_t total_tiles = (batch_slice_num_pages + tile_granularity - 1) / tile_granularity;
    constexpr uint32_t num_packets = (total_tiles + num_links - 1) / num_links;
    constexpr uint32_t tiles_per_slice = batch_slice_num_pages / num_links;

    for (uint32_t b = 0; b < num_batches; b++) {
        for (uint32_t i = 0; i < ring_size - 1; i++) {  // Don't reduce on the first slice
            // Initialize binary operations - use the same constants consistently
            binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
            add_tiles_init(input_cb_id, intermediate_cb, false);

            // Wait for input data once before beginning processing
            for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
                uint32_t to_process = std::min(tile_granularity, tiles_per_slice - packet_id * tile_granularity);
                cb_wait_front(input_cb_id, to_process);
                // Reserve output space once before processing
                cb_wait_front(intermediate_cb, to_process);
                cb_reserve_back(output_cb, to_process);
                acquire_dst();
                for (uint32_t tile_id = 0; tile_id < to_process; tile_id++) {
                    add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                    pack_tile(tile_id, output_cb);
                }
                release_dst();
                cb_pop_front(input_cb_id, to_process);
                cb_pop_front(intermediate_cb, to_process);
                cb_push_back(output_cb, to_process);
            }
        }
    }
}
}  // namespace NAMESPACE
