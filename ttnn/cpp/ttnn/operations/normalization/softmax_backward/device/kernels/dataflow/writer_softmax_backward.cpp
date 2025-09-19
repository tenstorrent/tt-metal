// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(1);

    // Adjustable batch size - must match compute kernel
    constexpr uint32_t tiles_per_batch = 4;

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

    // Write output rows in batches
    for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        const uint32_t row_start_tile = (start_tile + row_idx) * num_tiles_per_row;

        // Process tiles in batches within each row
        for (uint32_t batch_start = 0; batch_start < num_tiles_per_row; batch_start += tiles_per_batch) {
            // Calculate batch size (handle remainder)
            const uint32_t current_batch_size = (batch_start + tiles_per_batch <= num_tiles_per_row)
                                                    ? tiles_per_batch
                                                    : (num_tiles_per_row - batch_start);

            // Wait for compute to produce this batch
            cb_wait_front(out_cb_id, current_batch_size);
            const uint32_t l1_read_addr = get_read_ptr(out_cb_id);

            // Write tiles in this batch
            for (uint32_t i = 0; i < current_batch_size; ++i) {
                const uint32_t tile_idx = batch_start + i;
                noc_async_write(
                    l1_read_addr + i * out_tile_size,
                    output_accessor.get_noc_addr(row_start_tile + tile_idx),
                    out_tile_size);
            }

            // Wait for all writes in this batch to complete
            noc_async_write_barrier();

            // Pop this batch
            cb_pop_front(out_cb_id, current_batch_size);
        }
    }
}
