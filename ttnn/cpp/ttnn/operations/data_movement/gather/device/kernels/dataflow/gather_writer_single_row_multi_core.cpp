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
To improve performance writer kernel performs both writing as well as reading data.
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

    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
            const uint32_t current_index_tile_id = core_id + core_loop * total_number_of_cores;
            if (current_index_tile_id >= Wt_index) {
                break;
            }
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
            output_dfb.wait_front(one_tile);

            noc.async_write(
                output_dfb,
                output_tensor_dram,
                output_tensor_tile_size_bytes,
                {.offset_bytes = 0},
                {.page_id = h * Wt_index + current_index_tile_id});
            noc.async_write_barrier();
            output_dfb.pop_front(one_tile);

        }  // core_loop_count loop
    }  // h loop
}
