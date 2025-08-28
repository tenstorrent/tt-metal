// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_to_write = get_arg_val<uint32_t>(1);
    uint32_t start_output_stick_id = get_arg_val<uint32_t>(2);
    uint32_t grid_height = get_arg_val<uint32_t>(3);          // Original grid height (H_grid)
    uint32_t grid_width = get_arg_val<uint32_t>(4);           // Original grid width (W_grid)
    uint32_t batch_factor = get_arg_val<uint32_t>(5);         // Number of points batched per grid stick
    uint32_t start_grid_stick_id = get_arg_val<uint32_t>(6);  // Starting grid stick ID

    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t ntiles_c = get_compile_time_arg_val(2);

    constexpr auto dst_args = TensorAccessorArgs<3>();

    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);

    // For height-based batching, we need to write each output stick to the correct height position
    // Each output stick corresponds to a specific (batch, height, width) position in the final tensor

    for (uint32_t output_idx = 0; output_idx < num_sticks_to_write; output_idx++) {
        // Wait for ntiles_c pages in output CB (one full stick)
        cb_wait_front(cb_id_out0, ntiles_c);

        // Get base read address for this stick's data
        uint64_t base_l1_read_addr = get_read_ptr(cb_id_out0);

        // Calculate which grid stick and point within that stick this output corresponds to
        uint32_t absolute_output_stick_id = start_output_stick_id + output_idx;
        uint32_t relative_output_stick_id = absolute_output_stick_id - (start_grid_stick_id * batch_factor);

        uint32_t grid_stick_offset =
            relative_output_stick_id / batch_factor;                      // Which grid stick within this core's range
        uint32_t point_offset = relative_output_stick_id % batch_factor;  // Which point within that grid stick

        uint32_t grid_stick_id = start_grid_stick_id + grid_stick_offset;  // Absolute grid stick ID

        // Calculate the grid position of this stick
        uint32_t grid_batch = grid_stick_id / (grid_height * grid_width);
        uint32_t grid_stick_in_batch = grid_stick_id % (grid_height * grid_width);
        uint32_t grid_h_idx = grid_stick_in_batch / grid_width;  // Grid height index (0 to grid_height-1)
        uint32_t grid_w_idx = grid_stick_in_batch % grid_width;  // Grid width index (0 to grid_width-1)

        // Calculate final output position: height gets unbatched
        uint32_t final_h = grid_h_idx * batch_factor + point_offset;  // Unbatched height position
        uint32_t final_w = grid_w_idx;                                // Width position stays the same

        // Calculate final output stick ID in the tensor: (batch, final_h, final_w)
        uint32_t output_height = grid_height * batch_factor;  // Total height in output tensor
        uint32_t final_stick_id = grid_batch * (output_height * grid_width) + final_h * grid_width + final_w;

        // Write to the correct position
        uint64_t dst_noc_addr = s0.get_noc_addr(final_stick_id);
        noc_async_write(base_l1_read_addr, dst_noc_addr, output_stick_size);

        noc_async_write_barrier();

        // Pop the ntiles_c pages we just consumed
        cb_pop_front(cb_id_out0, ntiles_c);
    }
}
