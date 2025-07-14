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
    constexpr bool direction = get_compile_time_arg_val(8);

    uint32_t arg_idx = 0;
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);

    uint32_t tiles_read = (link * batch_slice_num_pages / num_links);
    uint32_t tiles_to_read = (link + 1) * batch_slice_num_pages / num_links;

    // Initialize binary operations - use the same constants consistently
    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    for (uint32_t b = 0; b < num_batches; b++) {
        for (uint32_t i = 0; i < ring_size - 1; i++) {  // Don't reduce on the first slice
            if constexpr (!direction) {
                uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                tiles_read += backwards_offset;
            }

            // Wait for input data once before beginning processing
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = 0;
                if constexpr (direction) {
                    num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                } else {
                    num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                }
                cb_wait_front(input_cb_id, tile_granularity);
                // Reserve output space once before processing
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
                tiles_read += num_pages_to_read;

                // Skip the tiles going the other direction
                if (tiles_read < tiles_to_read) {
                    num_pages_to_read = 0;
                    if constexpr (!direction) {
                        num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    } else {
                        num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                    }
                    tiles_read += num_pages_to_read;
                }
            }

            tiles_read = (link * batch_slice_num_pages / num_links);
        }
    }
}
}  // namespace NAMESPACE
