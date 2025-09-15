// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

constexpr uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

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
    constexpr uint32_t num_total_reduction_steps = get_compile_time_arg_val(8);
    constexpr uint32_t dim = get_compile_time_arg_val(9);
    constexpr uint32_t num_pages_per_slice = get_compile_time_arg_val(10);

    uint32_t arg_idx = 0;
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);

    uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
    uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;
    DPRINT << "tiles read: " << (uint32_t)tiles_read << " tiles to read: " << (uint32_t)tiles_to_read << "\n";
    uint32_t num_packets = div_up(tiles_to_read - tiles_read, tile_granularity);
    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    uint32_t running_batch_tile_offset = 0;
    for (uint32_t b = 0; b < num_batches; b++) {
        for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
            // Initialize binary operations - use the same constants consistently

            // Wait for input data once before beginning processing
            for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
                cb_wait_front(input_cb_id, tile_granularity);
                // Reserve output space once before processing
                cb_wait_front(intermediate_cb, tile_granularity);
                cb_reserve_back(output_cb, tile_granularity);
                acquire_dst();
                for (uint32_t tile_id = 0; tile_id < tile_granularity; tile_id++) {
                    if constexpr (dim == 3) {
                        add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                    } else {
                        uint32_t global_tile_index = tiles_read + packet_id * tile_granularity + tile_id;
                        uint32_t batch_tile_index = global_tile_index - running_batch_tile_offset;
                        uint32_t slice_index = batch_tile_index / (num_pages_per_slice / ring_size);
                        uint32_t tile_index_in_slice = batch_tile_index % (num_pages_per_slice / ring_size);
                        uint32_t intermediate_tile_index =
                            slice_index * (num_pages_per_slice / ring_size) + tile_index_in_slice;
                        add_tiles(input_cb_id, intermediate_cb, tile_id, intermediate_tile_index, tile_id);
                    }
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
}  // namespace NAMESPACE
