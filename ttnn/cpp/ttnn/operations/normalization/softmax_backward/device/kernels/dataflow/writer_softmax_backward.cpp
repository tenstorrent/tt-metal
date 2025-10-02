// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>
#include <tt-metalium/constants.hpp>

void kernel_main() {
    // Compile time args
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(0);

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    // Set up data movement for output tensor
    const InterleavedAddrGenFast<false> output_addrg = {
        .bank_base_address = output_addr,
        .page_size = tt::constants::TILE_HW * sizeof(uint16_t),
        .data_format = DataFormat::Float16_b};

    // For simplicity, assume we're processing one tile at a time
    // In a more complete implementation, this would handle multiple tiles per row

    // Write output tiles
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint32_t current_tile = start_tile + tile_idx;

        // Wait for compute to produce output
        cb_wait_front(out_cb_id, 1);
        uint32_t l1_read_addr = get_read_ptr(out_cb_id);

        // Write tile to output
        noc_async_write_tile(current_tile, output_addrg, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(out_cb_id, 1);
    }
}
