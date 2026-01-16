// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

/**
 * TopK Multicore Writer Kernel - Final Core Output to DRAM
 *
 * This kernel runs exclusively on the final aggregation core and handles the
 * final step of writing globally optimal TopK results to DRAM. It receives
 * the final aggregated and sorted results from the final compute kernel and
 * writes them to the output tensors in the proper format.
 *
 * Responsibilities:
 * 1. Read final TopK results from the final compute kernel
 * 2. Write TopK values tensor to DRAM in interleaved format
 * 3. Write TopK indices tensor to DRAM in interleaved format
 * 4. Ensure proper tile ordering for correct tensor layout
 *
 * Output Format:
 * - Values: Final TopK values in descending/ascending order as requested
 * - Indices: Corresponding original indices of the TopK values
 * - Layout: Standard interleaved tensor format for efficient host access
 */
void kernel_main() {
    // Runtime arguments - DRAM addresses for output tensors
    uint32_t dst_addr0 = get_arg_val<uint32_t>(0);  // DRAM address for TopK values output tensor
    uint32_t dst_addr1 = get_arg_val<uint32_t>(1);  // DRAM address for TopK indices output tensor

    // Compile-time configuration
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(0);      // Final values circular buffer
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(1);  // Final indices circular buffer
    constexpr uint32_t Ht = get_compile_time_arg_val(2);                   // Height tiles to write
    constexpr uint32_t Kt = get_compile_time_arg_val(3);                   // TopK tiles per height row

    // DRAM tensor accessor configuration for output writing
    constexpr auto interleaved_accessor0_args = TensorAccessorArgs<4>();
    constexpr auto interleaved_accessor1_args =
        TensorAccessorArgs<interleaved_accessor0_args.next_compile_time_args_offset()>();

    // Memory transfer configuration
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes_values = get_tile_size(values_cb_index);
    const uint32_t tile_bytes_ind = get_tile_size(output_ind_cb_index);

    // Initialize DRAM tensor accessors for interleaved output format
    const auto interleaved_accessor0 = TensorAccessor(interleaved_accessor0_args, dst_addr0, tile_bytes_values);
    const auto interleaved_accessor1 = TensorAccessor(interleaved_accessor1_args, dst_addr1, tile_bytes_ind);

    // MAIN OUTPUT LOOP: Write final TopK results to DRAM
    // Process each height row sequentially, writing Kt tiles of TopK results
    for (uint32_t j = 0; j < Ht; ++j) {  // For each height row
        // STEP 1: WRITE TopK VALUES TO DRAM
        // Write the final globally optimal TopK values for this height row
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(values_cb_index, onetile);  // Wait for final compute kernel output
            uint32_t l1_read_addr = get_read_ptr(values_cb_index);

            // Write tile to DRAM using interleaved layout
            // Tile index calculation: j * Kt + i ensures proper row-major ordering
            noc_async_write_tile(j * Kt + i, interleaved_accessor0, l1_read_addr);
            noc_async_write_barrier();  // Ensure write completes before buffer cleanup
            cb_pop_front(values_cb_index, onetile);
        }

        // STEP 2: WRITE TopK INDICES TO DRAM
        // Write the corresponding original indices for the TopK values
        for (uint32_t i = 0; i < Kt; ++i) {
            cb_wait_front(output_ind_cb_index, onetile);  // Wait for final compute kernel output
            uint32_t l1_read_addr = get_read_ptr(output_ind_cb_index);

            // Write tile to DRAM using interleaved layout
            // Same tile indexing as values to maintain correspondence
            noc_async_write_tile(j * Kt + i, interleaved_accessor1, l1_read_addr);
            noc_async_write_barrier();  // Ensure write completes before buffer cleanup
            cb_pop_front(output_ind_cb_index, onetile);
        }
    }
}
