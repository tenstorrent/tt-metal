// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include <cstdint>

/*
To improve performance writer kernel performs both writing as well as reading data.
    * Reads input tensor values from DRAM to L1.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(2);
    constexpr bool output_tensor_is_dram = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(5);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(6);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(7);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(8);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(9);

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_dram = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Output tensor config
    constexpr uint32_t output_tensor_tile_size_bytes = get_tile_size(output_tensor_cb_index);
    constexpr DataFormat output_tensor_data_format = get_dataformat(output_tensor_cb_index);
    const InterleavedAddrGenFast<output_tensor_is_dram> output_tensor_dram = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = output_tensor_tile_size_bytes,
        .data_format = output_tensor_data_format};

    const auto start_tile_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    uint32_t current_index_tile_id = start_tile_id;

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            // Read input data
            for (uint32_t w = 0; w < Wt_input; w++) {
                cb_reserve_back(input_tensor_cb_index, one_tile);
                const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
                noc_async_read_tile(h * Wt_input + w, input_tensor_dram, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(input_tensor_cb_index, one_tile);
            }  // Wt_input loop

            // Write output data
            cb_wait_front(output_tensor_cb_index, one_tile);

            const uint32_t l1_write_addr_output = get_read_ptr(output_tensor_cb_index);
            noc_async_write_tile(h * Wt_index + current_index_tile_id, output_tensor_dram, l1_write_addr_output);
            noc_async_write_barrier();
            cb_pop_front(output_tensor_cb_index, one_tile);

            current_index_tile_id += total_number_of_cores;
        }  // core_loop_count loop
    }  // h loop
}
