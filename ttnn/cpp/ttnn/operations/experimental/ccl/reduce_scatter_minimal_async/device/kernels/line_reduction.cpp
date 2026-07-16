// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    // Define all compile-time arguments at the beginning
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(1);
    constexpr uint32_t output_cb = get_compile_time_arg_val(2);
    constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
    constexpr uint32_t input_tensor_B = get_compile_time_arg_val(4);
    constexpr uint32_t slice_C = get_compile_time_arg_val(5);

    uint32_t arg_idx = 0;
    const uint32_t num_total_reduction_steps = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    DataflowBuffer cb_input(input_cb_id);
    DataflowBuffer cb_intermediate(intermediate_cb);
    DataflowBuffer cb_output(output_cb);

    binary_op_init_common(input_cb_id, intermediate_cb, output_cb);
    add_tiles_init(input_cb_id, intermediate_cb, false);

    for (uint32_t b = 0; b < input_tensor_B; b++) {
        for (uint32_t i = 0; i < num_total_reduction_steps; i++) {  // Don't reduce on the first slice
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_input.wait_front(tile_granularity);
                    cb_intermediate.wait_front(tile_granularity);

                    tile_regs_acquire();
                    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                        add_tiles(input_cb_id, intermediate_cb, tile_id, tile_id, tile_id);
                    }
                    tile_regs_commit();

                    cb_input.pop_front(tile_granularity);
                    cb_intermediate.pop_front(tile_granularity);

                    cb_output.reserve_back(tile_granularity);
                    tile_regs_wait();
                    for (uint32_t tile_id = 0; tile_id < num_pages_to_read; tile_id++) {
                        pack_tile(tile_id, output_cb);
                    }
                    tile_regs_release();
                    cb_output.push_back(tile_granularity);

                    tiles_read += num_pages_to_read;
                }
            }
        }
    }
}
