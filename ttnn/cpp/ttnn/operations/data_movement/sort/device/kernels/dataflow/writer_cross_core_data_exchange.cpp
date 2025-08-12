// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "cross_core_data_exchange_common.hpp"
#include "sort_dataflow_common.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(0);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_peer_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t physical_core_lookup_table_cb_index =
        get_compile_time_arg_val(5);  // unused - for future improvements
    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(10);          // unused - for future improvements
    const uint32_t sem_exchange_addr = get_semaphore(get_compile_time_arg_val(11));  // unused - for future improvements
    constexpr bool is_32_bit_data = get_compile_time_arg_val(12) == 1;

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_pair_start = core_id * number_of_pairs_processed_by_each_core;
    const uint16_t processing_pair_end = processing_pair_start + number_of_pairs_processed_by_each_core;

    // Output tensor config
    const uint32_t value_tensor_tile_size_bytes = get_tile_size(value_tensor_cb_index);
    const DataFormat value_tensor_data_format = get_dataformat(value_tensor_cb_index);
    const InterleavedAddrGenFast<value_tensor_is_dram> output_tensor_accessor = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = value_tensor_tile_size_bytes,
        .data_format = value_tensor_data_format};

    for (uint32_t h = 0; h < Ht; h++) {
        // Generate input index tiles
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            } else {
                generate_index_tile<uint16_t>(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            }
        }  // w loop

        // Write value tensor to DRAM
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_wait_front(value_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_val = get_read_ptr(value_tensor_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;

            noc_async_write_tile(tile_offset, output_tensor_accessor, l1_write_addr_val);
            noc_async_write_barrier();

            cb_pop_front(value_tensor_cb_index, one_tile);
        }  // Wt loop
    }  // h loop
    cb_push_back(physical_core_lookup_table_cb_index, one_tile);
}
