// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input value data from DRAM and writes it to L1 circular buffer.
    * Write processed index data from L1 to DRAM.

Writer:
    * Generates index input data and writes it to L1 circular buffer.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t total_number_of_cores = get_compile_time_arg_val(4);
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(5);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(6);

    constexpr auto input_tensor_args = TensorAccessorArgs<7>();
    constexpr auto index_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    // Input tensor config
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t tile_size_bytes = get_tile_size(input_tensor_cb_index);
    const auto interleaved_accessor0 = TensorAccessor(input_tensor_args, input_tensor_buffer_addr, tile_size_bytes);

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const auto interleaved_accessor1 =
        TensorAccessor(index_tensor_args, index_tensor_buffer_addr, index_tensor_output_tile_size_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb_input(input_tensor_cb_index);
    experimental::CircularBuffer cb_index_out(index_tensor_output_cb_index);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores +
                           get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();

        // Read input value data
        for (uint32_t w = 0; w < Wt; w++) {
            cb_input.reserve_back(one_tile);
            noc.async_read(
                interleaved_accessor0, cb_input, tile_size_bytes, {.page_id = h * Wt + w}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_input.push_back(one_tile);
        }  // Wt loop

        // Write output index data
        for (uint32_t w = 0; w < Wt; w++) {
            cb_index_out.wait_front(one_tile);
            noc.async_write(
                cb_index_out, interleaved_accessor1, index_tensor_output_tile_size_bytes, {}, {.page_id = h * Wt + w});
            noc.async_write_barrier();
            cb_index_out.pop_front(one_tile);
        }  // Wt loop
    }  // core_loop_count loop
}
