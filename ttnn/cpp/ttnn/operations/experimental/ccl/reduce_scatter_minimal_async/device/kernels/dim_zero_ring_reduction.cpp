// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("cb_input_id");
    constexpr uint32_t intermediate_cb = get_named_compile_time_arg_val("cb_interm_id");
    constexpr uint32_t output_cb = get_named_compile_time_arg_val("cb_compute_output_id");
    constexpr uint32_t tile_granularity = get_named_compile_time_arg_val("tile_granularity");
    constexpr uint32_t ring_size = get_named_compile_time_arg_val("ring_size");
    constexpr uint32_t slice_B = get_named_compile_time_arg_val("slice_B");

    uint32_t arg_idx = 0;
    uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    // Initialize binary operations - use the same constants consistently.
    // Hardware startup stays with the kernel; the accumulator owns only the op-level init.
    // (upstream renamed binary_op_init_common -> compute_kernel_hw_startup + per-op add_init.)
    compute_kernel_hw_startup(input_cb_id, intermediate_cb, output_cb);

    // Arm once: hoists add_init out of the chunk loop and asserts tile_granularity fits DEST.
    auto acc = compute_kernel_lib::BlockAccumulate::arm(input_cb_id, intermediate_cb, output_cb, tile_granularity);

    // Don't reduce on the first slice
    for (uint32_t i = 0; i < ring_size - 1; i++) {
        for (uint32_t b = 0; b < slice_B; b++) {
            uint32_t tiles_read = start_tiles_read;
            uint32_t tiles_to_read = start_tiles_to_read;

            if (!direction) {
                uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                tiles_read += backwards_offset;
            }

            // Interleave the two directions over one slice: this worker takes every other chunk, and
            // steps over the chunks belonging to the opposite direction without touching the CBs.
            while (tiles_read < tiles_to_read) {
                uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                uint32_t num_pages_to_read = 0;
                if (direction) {
                    num_pages_to_read = std::min(tiles_remaining_to_read / 2, tile_granularity);
                } else {
                    num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);
                }

                acc.run(num_pages_to_read);
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
