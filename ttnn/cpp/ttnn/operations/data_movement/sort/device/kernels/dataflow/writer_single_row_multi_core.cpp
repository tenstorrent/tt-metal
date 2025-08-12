// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t coordinator_core_physical_coord_x = get_arg_val<uint32_t>(2);
    const uint32_t coordinator_core_physical_coord_y = get_arg_val<uint32_t>(3);
    const uint32_t coordinator_to_cores_semaphore_id = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t cores_to_coordinator_semaphore_id = get_semaphore(get_arg_val<uint32_t>(5));

    // Compile time args
    constexpr uint32_t input_tensor_output_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(1);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool index_tensor_is_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t Ht = get_compile_time_arg_val(5);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(6);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(7);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(8);
    constexpr uint32_t number_of_available_cores = get_compile_time_arg_val(9);

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_output_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_output_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_addr_gen = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_output_cb_index);
    const InterleavedAddrGenFast<index_tensor_is_dram> index_tensor_addr_gen = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

    // Semaphore setup
    const uint64_t coordinator_core_addr = get_noc_addr(
        coordinator_core_physical_coord_x, coordinator_core_physical_coord_y, cores_to_coordinator_semaphore_id);

    for (uint32_t h = 0; h < Ht; h++) {
        // Get core start value
        const uint32_t core_start =
            get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Processing each row
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }

        for (uint32_t stage = 1; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);

                uint16_t pair_id = 0;
                uint32_t processing_pair_id = core_start;
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        if (pair_id == processing_pair_id) {
                            // Get indexes of tiles to compare
                            const uint32_t left_tile_id = i;
                            const uint32_t right_tile_id = j;

                            // Save index data
                            cb_wait_front(index_tensor_output_cb_index, one_tile);
                            const uint32_t l1_write_addr_index_output_tensor_cb_i =
                                get_read_ptr(index_tensor_output_cb_index);
                            noc_async_write_tile(
                                h * Wt + left_tile_id, index_tensor_addr_gen, l1_write_addr_index_output_tensor_cb_i);
                            noc_async_write_barrier();
                            cb_pop_front(index_tensor_output_cb_index, one_tile);

                            cb_wait_front(index_tensor_output_cb_index, one_tile);
                            const uint32_t l1_write_addr_index_output_tensor_cb_j =
                                get_read_ptr(index_tensor_output_cb_index);
                            noc_async_write_tile(
                                h * Wt + right_tile_id, index_tensor_addr_gen, l1_write_addr_index_output_tensor_cb_j);
                            noc_async_write_barrier();
                            cb_pop_front(index_tensor_output_cb_index, one_tile);

                            // Save output data
                            cb_wait_front(input_tensor_output_cb_index, one_tile);
                            const uint32_t l1_write_addr_output_tensor_cb_i =
                                get_read_ptr(input_tensor_output_cb_index);
                            noc_async_write_tile(
                                h * Wt + left_tile_id, input_tensor_addr_gen, l1_write_addr_output_tensor_cb_i);
                            noc_async_write_barrier();
                            cb_pop_front(input_tensor_output_cb_index, one_tile);

                            cb_wait_front(input_tensor_output_cb_index, one_tile);
                            const uint32_t l1_write_addr_output_tensor_cb_j =
                                get_read_ptr(input_tensor_output_cb_index);
                            noc_async_write_tile(
                                h * Wt + right_tile_id, input_tensor_addr_gen, l1_write_addr_output_tensor_cb_j);
                            noc_async_write_barrier();
                            cb_pop_front(input_tensor_output_cb_index, one_tile);

                            // Signalize readiness to the coordinator
                            noc_semaphore_inc(coordinator_core_addr, 1);

                            processing_pair_id += number_of_available_cores;
                        }  // if pair_id == processing_pair_id
                        pair_id++;
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop
    }  // h loop
}
