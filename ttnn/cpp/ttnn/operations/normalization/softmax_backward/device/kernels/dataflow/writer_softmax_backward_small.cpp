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

    // Common runtime args (shared across all cores)
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t total_num_rows = get_common_arg_val<uint32_t>(1);
    const uint32_t num_cores_x = get_common_arg_val<uint32_t>(2);
    const uint32_t num_cores_y = get_common_arg_val<uint32_t>(3);

    // Calculate work assignment for this core based on coordinates
    const uint32_t core_id_x = get_absolute_logical_x();
    const uint32_t core_id_y = get_absolute_logical_y();
    // Match factory's column-major indexing: core_idx = x * num_cores_y + y
    const uint32_t core_id = core_id_x * num_cores_y + core_id_y;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t rows_per_core = (total_num_rows + num_cores - 1) / num_cores;

    const uint32_t start_tile = core_id * rows_per_core;
    const uint32_t end_tile =
        ((start_tile + rows_per_core) < total_num_rows) ? (start_tile + rows_per_core) : total_num_rows;
    const uint32_t num_rows = (start_tile < total_num_rows) ? (end_tile - start_tile) : 0;

    // Get tile size
    const uint32_t out_tile_size = get_tile_size(out_cb_id);

    // Create tensor accessor
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_tile_size);

    // Early exit if this core has no work
    if (num_rows == 0) {
        return;
    }

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
