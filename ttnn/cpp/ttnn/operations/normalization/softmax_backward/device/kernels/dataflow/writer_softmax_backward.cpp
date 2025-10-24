// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(1);

    // Set up tensor accessor
    constexpr auto output_args = TensorAccessorArgs<2>();

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_rows = get_arg_val<uint32_t>(rt_args_idx++);

    // Get tile size
    const uint32_t out_tile_size = get_tile_size(out_cb_id);

    // Create tensor accessor
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_tile_size);

    // Write output rows
    for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Wait for compute to produce all tiles in the row
        cb_wait_front(out_cb_id, num_tiles_per_row);
        uint32_t l1_read_addr = get_read_ptr(out_cb_id);

        // Write all tiles in the row
        for (uint32_t w = 0; w < num_tiles_per_row; ++w) {
            noc_async_write(
                l1_read_addr + w * out_tile_size, output_accessor.get_noc_addr(row_start_tile + w), out_tile_size);
        }
        noc_async_write_barrier();
        cb_pop_front(out_cb_id, num_tiles_per_row);
    }
}
