// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "../scatter_common.hpp"

void kernel_main() {
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool src_tensor_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t input_tensor_addr = get_compile_time_arg_val(3);
    constexpr uint32_t index_tensor_addr = get_compile_time_arg_val(4);
    constexpr uint32_t src_tensor_addr = get_compile_time_arg_val(5);
    constexpr uint32_t input_tensor_cb = get_compile_time_arg_val(6);
    constexpr uint32_t index_tensor_cb = get_compile_time_arg_val(7);
    constexpr uint32_t src_tensor_cb = get_compile_time_arg_val(8);
    constexpr uint32_t output_tensor_cb = get_compile_time_arg_val(9);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(10);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(11);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(12);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(13);

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_addr_gtor = {
        .bank_base_address = input_tensor_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Index tensor config
    constexpr uint32_t index_tensor_tile_size_bytes = get_tile_size(index_tensor_cb);
    constexpr DataFormat index_tensor_data_format = get_dataformat(index_tensor_cb);
    const InterleavedAddrGenFast<index_tensor_is_dram> index_tensor_addr_gtor = {
        .bank_base_address = index_tensor_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    constexpr uint32_t src_tensor_tile_size_bytes = get_tile_size(src_tensor_cb_index);
    constexpr DataFormat src_tensor_data_format = get_dataformat(src_tensor_cb_index);
    const InterleavedAddrGenFast<src_tensor_is_dram> output_tensor_addr_gtor = {
        .bank_base_address = src_tensor_addr,
        .page_size = src_tensor_tile_size_bytes,
        .data_format = src_tensor_data_format};

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Read input data
        for (uint32_t w = 0; w < Wt_input; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            noc_async_read_tile(h * Wt_input + w, input_tensor_dram, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);
        }

        // Write output data
        for (uint32_t w = 0; w < Wt_index; w++) {
            cb_wait_front(output_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_output = get_read_ptr(output_tensor_cb_index);
            noc_async_write_tile(h * Wt_index + w, output_tensor_dram, l1_write_addr_output);
            noc_async_write_barrier();
            cb_pop_front(output_tensor_cb_index, one_tile);
        }
    }
}
