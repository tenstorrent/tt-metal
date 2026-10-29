// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"

#include <cstdint>

/*
To improve performance of both reader and writer kernels the work has been split so that they both prepare input and
save output data.

Reader:
    * Reads input index tensor data from DRAM and writes it to L1 dataflow buffer.
    * Performs calculation on indexes to get the correct value from input tensor indexes.
    * Writes values to output L1 dataflow buffer.

Writer:
    * Reads input tensor values from DRAM to L1.
    * Write output values from L1 to DRAM.
*/
void kernel_main() {
    // Runtime args
    const uint32_t core_loop_count = get_arg(args::core_loop_count);
    const uint32_t core_id = get_arg(args::core_id);

    // Compile time args
    constexpr bool input_tensor_is_dram = get_arg(args::input_tensor_is_dram);
    constexpr bool output_tensor_is_dram = get_arg(args::output_tensor_is_dram);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt_input = get_arg(args::Wt_input);
    constexpr uint32_t Wt_index = get_arg(args::Wt_index);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr uint32_t compute_with_storage_grid_size_x = get_arg(args::compute_with_storage_grid_size_x);
    constexpr uint32_t compute_with_storage_grid_size_y = get_arg(args::compute_with_storage_grid_size_y);

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    const auto input_tensor_dram = TensorAccessor(tensor::input_tensor);

    // Output tensor config
    const auto output_tensor_dram = TensorAccessor(tensor::output_tensor);

    // Tile size in bytes for input and output tensors
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(dfb::input_tensor);
    constexpr uint32_t output_tensor_tile_size_bytes = get_tile_size(dfb::output_tensor);

    Noc noc;
    DataflowBuffer input_dfb(dfb::input_tensor);
    DataflowBuffer output_dfb(dfb::output_tensor);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        // Calculate tile h coordinate
        const uint32_t h = core_loop * total_number_of_cores + core_id;

        // Read input data
        for (uint32_t w = 0; w < Wt_input; w++) {
            input_dfb.reserve_back(one_tile);
            noc.async_read(
                input_tensor_dram,
                input_dfb,
                input_tensor_tile_size_bytes,
                {.page_id = h * Wt_input + w},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            input_dfb.push_back(one_tile);
        }  // Wt_input loop

        // Write output data
        for (uint32_t w = 0; w < Wt_index; w++) {
            output_dfb.wait_front(one_tile);
            noc.async_write(
                output_dfb,
                output_tensor_dram,
                output_tensor_tile_size_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt_index + w});
            noc.async_write_barrier();
            output_dfb.pop_front(one_tile);
        }  // Wt_index loop
    }  // core_loop_count loop
}
