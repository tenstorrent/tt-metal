// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>

void kernel_main() {
    // Compile time args
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(0);

    // Set up tensor accessor
    constexpr auto output_args = TensorAccessorArgs<1>();

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    // Get tile size
    const uint32_t out_tile_size = get_tile_size(out_cb_id);

    // Create tensor accessor
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_tile_size);

    // Write output tiles
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint32_t current_tile = start_tile + tile_idx;

        // Wait for compute to produce output
        cb_wait_front(out_cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(out_cb_id);

        // Write tile to output
        noc_async_write(l1_read_addr, output_accessor.get_noc_addr(current_tile), out_tile_size);
        noc_async_write_barrier();
        cb_pop_front(out_cb_id, 1);
    }
}
