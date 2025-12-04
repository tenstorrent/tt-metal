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
    constexpr uint32_t slice_B = get_compile_time_arg_val(5);

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    // Initialize binary operations - use the same constants consistently
    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    // Don't reduce on the first slice
    for (uint32_t i = 0; i < ring_size - 1; i++) {
        for (uint32_t b = 0; b < slice_B; b++) {
            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            if (!direction) {
                uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                tiles_read += backwards_offset;
            }

            // Wait for input data once before beginning processing
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t num_pages_to_read = 0;
                if (direction) {
                    num_pages_to_read = std::min(tiles_remaining_to_read / 2, tile_granularity);
                } else {
                    num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);
                }
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
                tiles_read += num_pages_to_read;

                // Skip the tiles going the other direction
                tiles_remaining_to_read = tiles_to_read - tiles_read;
                if (tiles_remaining_to_read > 0) {
                    num_pages_to_read = 0;
                    if (!direction) {
                        num_pages_to_read = std::min(tiles_remaining_to_read / 2, tile_granularity);
                    } else {
                        num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);
                    }
                    tiles_read += num_pages_to_read;
                }
            }
        }
    }
}
}  // namespace NAMESPACE
