// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t batch_size = get_compile_time_arg_val(0);      // Number of batches
    constexpr uint32_t input_height = get_compile_time_arg_val(1);    // Height of input tensor
    constexpr uint32_t input_width = get_compile_time_arg_val(2);     // Width of input tensor
    constexpr uint32_t stride_height = get_compile_time_arg_val(3);   // Vertical stride
    constexpr uint32_t stride_width = get_compile_time_arg_val(4);    // Horizontal stride
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(5);    // Size of each stick in bytes
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(6);  // Number of tiles per row
    constexpr uint32_t element_size = get_compile_time_arg_val(7);    // Size of each element in bytes
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(8);       // Input compute buffer ID

    // Runtime arguments
    uint32_t dst_addr = get_arg_val<uint32_t>(0);          // Destination address in DRAM
    uint32_t start_block_id = get_arg_val<uint32_t>(1);    // Starting block ID for processing
    uint32_t num_blocks = get_arg_val<uint32_t>(2);        // Number of blocks to process
    uint32_t nblocks_per_core = get_arg_val<uint32_t>(3);  // Number of blocks per core

    // Constants for tile dimensions
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    // dump(stick_nbytes);

    // Initialize DRAM address generator
    constexpr auto tensor_args = TensorAccessorArgs<1, 1, 0>();  // Hard-code DRAM
    const auto d = TensorAccessor(tensor_args, dst_addr, stick_nbytes);

    // Pre-calculate output dimensions and patch size - moved outside loops
    const uint32_t OH = input_height / stride_height;                   // Output height
    const uint32_t OW = input_width / stride_width;                     // Output width
    const uint32_t patch_size = stride_height * stride_width;           // Elements per patch
    const uint32_t elements_per_batch = OH * OW * patch_size;           // Elements in one batch
    const uint32_t elements_per_row = input_width * patch_size;         // Elements in output row
    const uint32_t sticks_per_output_row = stride_height * input_width;  // Sticks per output row

    // Pre-calculated indices for faster processing - moved outside loops for efficiency
    uint32_t output_row = 0;
    uint32_t input_batch = 0;
    uint32_t input_row = 0;
    uint32_t output_col = 0;

    // Process each block sequentially with pre-calculated parameters
    for (uint32_t nblock = start_block_id; nblock < start_block_id + num_blocks; nblock++) {
        // Wait for data to be available in the circular buffer
        cb_wait_front(cb_id_in1, ntiles_per_row);
        uint32_t l1_read_addr = get_read_ptr(cb_id_in1);

        // Calculate the global linear output index for this block
        uint32_t global_linear_output_idx = nblock * ntiles_per_row;

        // Process each tile in the current block using the global index
        for (uint32_t j = 0; j < ntiles_per_row; j++) {
            uint32_t current_global_idx = global_linear_output_idx + j;

            // Use integer arithmetic to calculate batch, tile, and position indices
            uint32_t batch_idx = current_global_idx / elements_per_batch;
            uint32_t remaining_idx = current_global_idx % elements_per_batch;

            uint32_t output_row_idx = remaining_idx / elements_per_row;
            uint32_t remaining_in_row = remaining_idx % elements_per_row;

            uint32_t output_col_idx = remaining_in_row / patch_size;
            uint32_t patch_element_idx = remaining_in_row % patch_size;

            // Map patch element index to original spatial coordinates
            uint32_t patch_h = patch_element_idx / stride_width;
            uint32_t patch_w = patch_element_idx % stride_width;

            // Calculate actual input position
            uint32_t input_row_actual = output_row_idx * stride_height + patch_h;
            uint32_t input_col_actual = output_col_idx * stride_width + patch_w;

            // Calculate final linear output index
            uint32_t final_output_idx =
                batch_idx * input_height * input_width + input_row_actual * input_width + input_col_actual;

            // Get destination address and perform write operation
            uint64_t dst_noc_addr = d.get_noc_addr(final_output_idx);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_nbytes);

            // Advance to next tile in memory
            l1_read_addr += stick_nbytes;
        }

        // Wait for all writes to complete before proceeding
        noc_async_write_barrier();

        // Pop completed data from the circular buffer
        cb_pop_front(cb_id_in1, ntiles_per_row);
    }
}
