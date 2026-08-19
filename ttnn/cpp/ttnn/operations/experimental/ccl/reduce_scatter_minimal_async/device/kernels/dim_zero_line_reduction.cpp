// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t slice_B = get_compile_time_arg_val(4);

    uint32_t arg_idx = 0;
    const uint32_t num_total_reduction_steps = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    // Hardware startup stays with the kernel; the accumulator owns only the op-level init.
    // (upstream renamed binary_op_init_common -> compute_kernel_hw_startup + per-op add_init.)
    compute_kernel_hw_startup(input_cb_id, intermediate_cb, output_cb);

    // Arm once: hoists add_init out of the chunk loop and asserts tile_granularity fits DEST.
    auto acc = compute_kernel_lib::BlockAccumulate::arm(input_cb_id, intermediate_cb, output_cb, tile_granularity);

    for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
        for (uint32_t b = 0; b < slice_B; ++b) {
            uint32_t tiles_read = start_tiles_read;
            const uint32_t tiles_to_read = start_tiles_to_read;

            while (tiles_read < tiles_to_read) {
                const uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                acc.run(num_pages_to_read);
                tiles_read += num_pages_to_read;
            }
        }
    }
}
