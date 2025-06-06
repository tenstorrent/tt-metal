// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <tt-metalium/constants.hpp>

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_loop_count = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(3);
    constexpr bool output_tensor_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(6);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(7);

    constexpr uint32_t one_tile = 1;

    // Output tensor config
    const uint32_t output_tensor_output_tile_size_bytes = get_tile_size(output_tensor_cb_index);
    const DataFormat output_tensor_output_data_format = get_dataformat(output_tensor_cb_index);
    const InterleavedAddrGenFast<output_tensor_is_dram> output_tensor_addr_gen = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = output_tensor_output_tile_size_bytes,
        .data_format = output_tensor_output_data_format};

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        uint32_t currently_processed_output_tile = core_start;
        for (uint32_t index_loop = 0; index_loop < index_loop_count; index_loop++) {
            // Save output tile
            cb_wait_front(output_tensor_cb_index, one_tile);
            DPRINT << "WRITER: index_tile: " << U32(currently_processed_output_tile) << ENDL();
            const uint32_t l1_output_read_addr = get_read_ptr(output_tensor_cb_index);
            noc_async_write_tile(
                h * Wt_index + currently_processed_output_tile, output_tensor_addr_gen, l1_output_read_addr);
            noc_async_write_barrier();
            cb_pop_front(output_tensor_cb_index, one_tile);

            // Increment the output tile
            currently_processed_output_tile += number_of_available_cores;
        }
    }
}
