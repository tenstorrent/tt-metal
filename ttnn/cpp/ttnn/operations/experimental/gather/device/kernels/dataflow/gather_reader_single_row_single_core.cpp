// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_common.hpp"

#include "dataflow_api.h"
#include "debug/dprint.h"
#include <tt-metalium/constants.hpp>

#include <cstdint>

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input index tensor data from DRAM and writes it to L1 circular buffer.
    * Performs calculation on indexes to get the correct value from input tensor indexes.
    * Writes values to output L1 circular buffer.

Writer:
    * Reads input tensor values from DRAM to L1.
    * Write output values from L1 to DRAM.

--- Algorithm description ---

This sorting kernel implements gather operation functionality.

The algorithm Gathers values along an axis specified by dim from input tensor based on value indexes stored inside input
index tensor and writes them to output tensor. Input tensor and input index tensor must have the same number of
dimensions. It is also required that input_index_tensor.size(d) <= input_tensor.size(d) for all dimensions d != dim.
Output_tensor will have the same shape as index.

### Overview:
1. **Tile Initialization**:
    - Computation is performed always on dim = -1 - the tensors are transposed to have the last dimension as the
      dimension of interest.
    - Only the last dimension in the input tensor (the one that computation is done on) can be different in size than in
the input_index_tensor.
    - Wt_input represents the number of tiles in the last dimension of the input tensor.
    - Wt_index represents the number of tiles in the last dimension of the input index tensor.
    - A full row of tiles of input tensor (size `Wt_input`) is read from DRAM into L1 memory.
    - One tile in the ouput buffer is reserved for each tile in the input index tensor.

2. **Computation mechanism**:
    - Tiles from Wt_index are read from L1 memory one by one.
    - For each index tile the one ouput tile is reserved in the output buffer.
    - Algorithm iterates over the values in the index tile (`global_index`) - these values represents the indexes of the
values in the input tensor that should be gathered.
    - The values are read regardless of the datatype. The datatype of the tile determines read/write mechanism
automatically.

3. **Index mapping**:
    - Global index values (`global_index`) needs to be mapped to the corresponding indices in the input tensor tiles of
size `Wt_input`.
    - Mapping:
        - Determine the tile index (`tile_idx`) by dividing the global index by the constant `TILE_WIDTH`. This
identifies which tile the global index belongs to.
        - Compute the position within the local tile (`index_in_local_tile`) by taking the remainder of the global index
divided by `TILE_WIDTH`. This gives the offset within the tile.
        - Determine the row (`which_row`) and column (`which_col`) within the local tile:
           - `which_row` is calculated by dividing `index_in_local_tile` by `face_size`.
           - `which_col` is calculated by taking the remainder of `index_in_local_tile` divided by `face_size`.
        - Combine the above components to compute the final local index (`local_index`):
            - Multiply `tile_idx` by `TILE_HW` to account for the offset of the tile in the global space.
            - Add the row offset within the tile, calculated as `which_row * (face_size * face_size)`.
            - Add the depth offset within the tile, calculated as `k * face_size`.
            - Add the column offset within the tile, calculated as `which_col`.
            - Add the face offset, calculated as `i * (TILE_WIDTH * face_size)`.
        - This results in the final `local_index`, which is used to access the appropriate value within the local tile.

4. **Multicore Calculation**:
    - Multicore parallelism is enabled by assigning each row of tiles (`Wt_index`) to a separate core.
    - If the number of rows (`Ht`) exceeds the number of available cores, the workload is distributed such that some
cores process multiple rows.
    - This ensures efficient utilization of all cores and minimizes idle time during computation.

5. **Final Steps**:
    - Once the appropriate value is read from input tensor circular buffer they are placed in the output tensor L1.
    - The output data is then written back to DRAM.

### Example:
- Input: A 64x128 matrix, represented as 2x4 tiles: T_in0, T_in1, T_in2, T_in3
                                                    T_in4, T_in5, T_in6, T_in7
    - Wt_input = 4
- Input index tensor: 64x64 matrix, represented as 2x4 tiles: T_idx0, T_idx1
                                                              T_idx2, T_idx3
    - Wt_index = 2
- Output: A 64x64 matrix, represented as 2x4 tiles: T_out0, T_out1
                                                    T_out2, T_out3
    - Wt_index = 2 (the same as index)
- Dim: -1
0. Distributing workload across cores:
   - Core 0 processes T_in0, T_in1, T_in2, T_in3 / T_idx0, T_idx1 / T_out0, T_out1
   - Core 1 processes T_in4, T_in5, T_in6, T_in7 / T_idx2, T_idx3 / T_out2, T_out3
Calculation of each row:
  1. **Reading data**:
      - Reading whole Wt_input row: T_in0, T_in1, T_in2, T_in3.
      - Reading one tile at a time of index tensor: T_idx,
      - Reserving one tile in output tensor: T_out.
  2. **Gathering data**:
      - For each tile in index tensor:
        - Read the index values from the index tensor tile.
        - For each index value, calculate the corresponding local index in the input tensor tile.
        - Read the value from the input tensor tile using the calculated local index.
        - Write the gathered value to the output tensor tile at the reserved location.
  3. **Data Saving**:
      - Output tiles are saved from L1 to DRAM.
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
    const uint32_t tile_width_mask = tile_width - 1;

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

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        for (uint32_t w = 0; w < Wt_index; w++) {
            // Read index data
            cb_reserve_back(input_index_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_write_ptr(input_index_tensor_cb_index);
            noc_async_read_tile(h * Wt_index + w, input_index_tensor_dram, l1_write_addr_index);
            noc_async_read_barrier();
            cb_push_back(input_index_tensor_cb_index, one_tile);

            cb_wait_front(input_tensor_cb_index, Wt_input);
            cb_wait_front(input_index_tensor_cb_index, one_tile);
            cb_reserve_back(output_tensor_cb_index, one_tile);

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
                            const uint32_t index_in_local_tile = global_index & tile_width_mask;
                            const uint32_t which_row = index_in_local_tile >> __builtin_ctz(face_size);
                            const uint32_t which_col = index_in_local_tile & FACE_SIZE_MASK;
                            /* Equivalent to:
                            const uint32_t tile_idx = global_index / tile_width;
                            const uint32_t index_in_local_tile = global_index % tile_width;
                            const uint32_t which_row = index_in_local_tile / face_size;
                            const uint32_t which_col = index_in_local_tile % face_size;

                            Division is replaced with bit shift,
                            Modulo replaced with bitwise AND with mask.
                            */
                            const uint16_t local_index = tile_idx * (tile_width * tile_height) +
                                                         which_row * (face_size * face_size) + k * face_size +
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
            cb_push_back(output_tensor_cb_index, one_tile);
            cb_pop_front(input_index_tensor_cb_index, one_tile);
        }  // Wt loop
        cb_pop_front(input_tensor_cb_index, Wt_input);
    }  // core_loop_count loop
}
