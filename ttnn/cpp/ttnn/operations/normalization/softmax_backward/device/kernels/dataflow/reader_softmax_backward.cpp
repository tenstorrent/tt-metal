// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <dataflow_api.h>
#include <cstdint>
#include <tt-metalium/constants.hpp>

void kernel_main() {
    // Compile time args
    constexpr uint32_t src0_cb_id = get_compile_time_arg_val(0);  // softmax_output
    constexpr uint32_t src1_cb_id = get_compile_time_arg_val(1);  // upstream_grad
    constexpr uint32_t num_tiles_per_row = get_compile_time_arg_val(2);

    // Runtime args
    uint32_t rt_args_idx = 0;
    const uint32_t softmax_output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t upstream_grad_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_tile = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);

    // Set up data movement for softmax_output tensor
    const InterleavedAddrGenFast<true> softmax_output_addrg = {
        .bank_base_address = softmax_output_addr,
        .page_size = tt::constants::TILE_HW * sizeof(uint16_t),
        .data_format = DataFormat::Float16_b};

    // Set up data movement for upstream_grad tensor
    const InterleavedAddrGenFast<true> upstream_grad_addrg = {
        .bank_base_address = upstream_grad_addr,
        .page_size = tt::constants::TILE_HW * sizeof(uint16_t),
        .data_format = DataFormat::Float16_b};

    // Process tiles one by one for simplicity
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint32_t current_tile = start_tile + tile_idx;

        // Read one tile from softmax_output
        cb_reserve_back(src0_cb_id, 1);
        uint32_t l1_write_addr_src0 = get_write_ptr(src0_cb_id);
        noc_async_read_tile(current_tile, softmax_output_addrg, l1_write_addr_src0);
        noc_async_read_barrier();
        cb_push_back(src0_cb_id, 1);

        // Read one tile from upstream_grad
        cb_reserve_back(src1_cb_id, 1);
        uint32_t l1_write_addr_src1 = get_write_ptr(src1_cb_id);
        noc_async_read_tile(current_tile, upstream_grad_addrg, l1_write_addr_src1);
        noc_async_read_barrier();
        cb_push_back(src1_cb_id, 1);
    }
}
