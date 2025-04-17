// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    // Runtime args
    const uint32_t value_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(1);
    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(2);
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(6);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(7);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(8);

    // Output tensor config
    constexpr uint32_t one_tile = 1;
    const uint32_t value_tensor_tile_size_bytes = get_tile_size(value_tensor_cb_index);
    const DataFormat value_tensor_data_format = get_dataformat(value_tensor_cb_index);
    const InterleavedAddrGenFast<value_tensor_is_dram> interleaved_accessor0 = {
        .bank_base_address = value_tensor_buffer_addr,
        .page_size = value_tensor_tile_size_bytes,
        .data_format = value_tensor_data_format};

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_output_cb_index);
    const InterleavedAddrGenFast<index_tensor_is_dram> interleaved_accessor1 = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    // Move data from L1 to DRAMs
    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(value_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_val = get_read_ptr(value_tensor_cb_index);
            noc_async_write_tile(h * Wt + w, interleaved_accessor0, l1_write_addr_val);
            noc_async_write_barrier();
            cb_pop_front(value_tensor_cb_index, one_tile);
        }  // Wt loop

        // Index tensor
        for (uint32_t w = 0; w < Wt; w++) {
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_read_ptr(index_tensor_output_cb_index);
            noc_async_write_tile(h * Wt + w, interleaved_accessor1, l1_write_addr_index);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop
    }  // core_loop_count loop
}
