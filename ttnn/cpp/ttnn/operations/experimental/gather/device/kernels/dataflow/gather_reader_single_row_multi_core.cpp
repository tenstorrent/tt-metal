// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_common.hpp"

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <tt-metalium/constants.hpp>

#include <cstdint>
/*
This kernel implements a parallel gather operation along the last dimension (Wt_index) of the tensor, enabling support
for all tensor sizes without memory constraints.

--- Algorithm Description ---

1. **Parallel Row Processing**:
    - Each core is assigned a portion of the Wt_index dimension (the axis along which the gather is performed).
    - All rows (Ht) are processed sequentially, but within each row, tiles along Wt_index are distributed across
available cores.
    - If the number of tiles in Wt_index exceeds the number of cores, each core processes multiple tiles in a loop.

2. **Per-Core Work Assignment**:
    - Each core calculates its starting tile in Wt_index based on its absolute core coordinates.
    - The core processes its assigned tiles, then increments by the total number of cores to process additional tiles if
needed.

3. **Processing Steps**:
    - For each assigned Wt_index tile:
        - Read the input index tensor tile from DRAM into L1.
        - Reserve space for the corresponding output tile in L1.
        - For each input tensor tile (Wt_input):
            - Wait for the input tensor tile to be available in L1.
            - For each value in the input index tensor tile:
                - If the index points to a value in the current input tensor tile, process it:
                    - Calculate the local index within the tile.
                    - Read the value from the input tensor tile.
                    - Write the value to the output tensor tile at the correct position.
                - If not, skip to the next value.
            - Move to the next input tensor tile.
        - Push the completed output tile to the output buffer.
        - Pop the processed input index tensor tile from the buffer.
        - Move to the next assigned Wt_index tile (by incrementing by the total number of cores).

This approach enables parallel processing of the gather operation along the Wt_index dimension, maximizing core
utilization and supporting large tensors.
*/
void kernel_main() {
    // Runtime args
    const uint32_t input_index_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(1);
    const uint32_t tile_width = get_arg_val<uint32_t>(2);
    const uint32_t tile_height = get_arg_val<uint32_t>(3);

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
    const uint32_t TILE_WIDTH_MASK = tile_width - 1;

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
                                const uint32_t tile_idx = global_index >> __builtin_ctz(tile_width);

                                ASSERT(
                                    tile_idx <= Wt_input,
                                    "Index out of range. Index: {}, Max index: {}",
                                    global_index,
                                    Wt_input * tile_width);

                                if (tile_idx != wi) {
                                    // Index not in current input tile, skip
                                    count++;
                                    continue;
                                }

                                const uint32_t index_in_local_tile = global_index & TILE_WIDTH_MASK;
                                const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                                const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;

                                const uint16_t local_index = which_row * (face_size * face_size) + k * face_size +
                                                             which_col + i * (tile_width * face_size);

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
