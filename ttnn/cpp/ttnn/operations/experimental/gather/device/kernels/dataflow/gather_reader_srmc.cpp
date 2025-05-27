// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_common.hpp"

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <tt-metalium/constants.hpp>

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t input_index_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t output_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr bool input_index_tensor_is_dram = get_compile_time_arg_val(3);
    constexpr uint32_t Ht = get_compile_time_arg_val(4);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(5);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(6);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(7);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(8);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(9);

    constexpr uint32_t one_tile = 1;
    constexpr uint32_t TILE_WIDTH_MASK = tt::constants::TILE_WIDTH - 1;

    // Index tensor config
    constexpr uint32_t input_index_tensor_tile_size_bytes = get_tile_size(input_index_tensor_cb_index);
    constexpr DataFormat input_index_tensor_data_format = get_dataformat(input_index_tensor_cb_index);
    const InterleavedAddrGenFast<input_index_tensor_is_dram> input_index_tensor_dram = {
        .bank_base_address = input_index_tensor_buffer_addr,
        .page_size = input_index_tensor_tile_size_bytes,
        .data_format = input_index_tensor_data_format};

    // Dataformats size
    constexpr uint32_t input_tensor_data_format_size =
        get_tile_size(input_tensor_cb_index) / get_tile_hw(input_tensor_cb_index);
    constexpr uint32_t input_index_tensor_data_format_size =
        input_index_tensor_tile_size_bytes / get_tile_hw(input_index_tensor_cb_index);
    constexpr uint32_t output_tensor_data_format_size =
        get_tile_size(output_tensor_cb_index) / get_tile_hw(input_tensor_cb_index);

    const auto start_tile_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    uint32_t current_index_tile_id = start_tile_id;

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            // Read index data
            cb_reserve_back(input_index_tensor_cb_index, one_tile);

            const uint32_t l1_write_addr_index = get_write_ptr(input_index_tensor_cb_index);
            noc_async_read_tile(h * Wt_index + current_index_tile_id, input_index_tensor_dram, l1_write_addr_index);
            noc_async_read_barrier();

            cb_push_back(input_index_tensor_cb_index, one_tile);
            cb_wait_front(input_index_tensor_cb_index, one_tile);

            cb_reserve_back(output_tensor_cb_index, one_tile);

            for (uint32_t wi = 0; wi < Wt_input; wi++) {
                cb_wait_front(input_tensor_cb_index, one_tile);

                const uint32_t input_tensor_l1_read_addr = get_read_ptr(input_tensor_cb_index);
                const uint32_t input_index_tensor_l1_read_addr = get_read_ptr(input_index_tensor_cb_index);
                const uint32_t output_tensor_l1_read_addr = get_read_ptr(output_tensor_cb_index);

                uint32_t count = 0;
                constexpr uint32_t tile_faces = 2;
                constexpr uint32_t face_size = 16;
                constexpr uint32_t FACE_SIZE_MASK = face_size - 1;
                for (uint32_t i = 0; i < tile_faces; ++i) {
                    for (uint32_t j = 0; j < tile_faces; ++j) {
                        for (uint32_t k = 0; k < face_size; ++k) {
                            for (uint32_t l = 0; l < face_size; l++) {
                                // Read global index
                                const uint32_t global_index = get_value_from_tile(
                                    input_index_tensor_l1_read_addr, count, input_index_tensor_data_format_size);

                                // Calculate local index
                                const uint32_t tile_idx = global_index >> __builtin_ctz(tt::constants::TILE_WIDTH);

                                if (tile_idx != wi) {
                                    // Index not in current input tile, skip
                                    count++;
                                    continue;
                                }

                                const uint32_t index_in_local_tile = global_index & TILE_WIDTH_MASK;
                                const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                                const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                                const uint16_t local_index = which_row * (face_size * face_size) + k * face_size +
                                                             which_col + i * (tt::constants::TILE_WIDTH * face_size);

                                // Read value
                                const uint32_t value = get_value_from_tile(
                                    input_tensor_l1_read_addr, local_index, input_tensor_data_format_size);

                                // Write value to output
                                write_value_to_tile(
                                    output_tensor_l1_read_addr, count, output_tensor_data_format_size, value);
                                count++;
                            }  // l loop
                        }  // k loop
                    }  // j loop
                }  // i loop
                cb_pop_front(input_tensor_cb_index, one_tile);
            }  // wi loop
            cb_push_back(output_tensor_cb_index, one_tile);
            cb_pop_front(input_index_tensor_cb_index, one_tile);
            current_index_tile_id += total_number_of_cores;
        }  // core_loop_count loop
    }  // h loop
}
