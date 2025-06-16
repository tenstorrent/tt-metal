// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"
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

    const uint32_t num_packets = batch_slice_num_pages / tile_granularity / num_links;
    UNPACK(
        DPRINT << "num_packets: " << num_packets << ", batch_slice_num_pages: " << batch_slice_num_pages
               << ", tile_granularity: " << tile_granularity << ", num_links: " << num_links << "\t" << num_packets
               << " num_total_reduction_steps: " << num_total_reduction_steps << "\n");

    for (uint32_t b = 0; b < num_batches; b++) {
        for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
            // Initialize binary operations - use the same constants consistently
            binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
            add_tiles_init(input_cb_id, intermediate_cb, false);

            // Wait for input data once before beginning processing
            for (uint32_t packet_id = 0; packet_id < num_packets; packet_id++) {
                cb_wait_front(input_cb_id, tile_granularity);
                // Reserve output space once before processing
                cb_wait_front(intermediate_cb, tile_granularity);
                cb_reserve_back(output_cb, tile_granularity);
                acquire_dst();
                for (uint32_t tile_id = 0; tile_id < tile_granularity; tile_id++) {
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

    UNPACK(DPRINT << "REDUCTION DONE\n");
}
}  // namespace NAMESPACE
