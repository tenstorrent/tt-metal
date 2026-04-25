// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

#include "api/debug/dprint.h"

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
*/
void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t core_loop_count = get_arg_val<uint32_t>(2);
    const uint32_t core_id = get_arg_val<uint32_t>(3);

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
    constexpr auto input_tensor_args = TensorAccessorArgs<10>();
    constexpr auto output_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    const auto input_tensor_dram = TensorAccessor(input_tensor_args, input_tensor_buffer_addr);

    // Output tensor config
    const auto output_tensor_dram = TensorAccessor(output_tensor_args, output_tensor_buffer_addr);

    // Tile size in bytes for input and output tensors
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr uint32_t output_tensor_tile_size_bytes = get_tile_size(output_tensor_cb_index);

    experimental::Noc noc;
    experimental::CircularBuffer input_cb(input_tensor_cb_index);
    experimental::CircularBuffer output_cb(output_tensor_cb_index);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores + core_id;

        // Read input data
        for (uint32_t w = 0; w < Wt_input; w++) {
            input_cb.reserve_back(one_tile);
            noc.async_read(
                input_tensor_dram,
                input_cb,
                input_tensor_tile_size_bytes,
                {.page_id = h * Wt_input + w},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_cb.push_back(one_tile);
        }  // Wt_input loop

        // Write output data
        for (uint32_t w = 0; w < Wt_index; w++) {
            output_cb.wait_front(one_tile);
            noc.async_write(
                output_cb,
                output_tensor_dram,
                output_tensor_tile_size_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt_index + w});
            noc.async_write_barrier();
            output_cb.pop_front(one_tile);
        }  // Wt_index loop
    }  // core_loop_count loop
}
