// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt-metalium/constants.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t input_width = get_compile_time_arg_val(0);            // Width of input tensor
    constexpr uint32_t stride_height = get_compile_time_arg_val(1);          // Vertical stride for fold
    constexpr uint32_t stride_width = get_compile_time_arg_val(2);           // Horizontal stride for fold
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(3);           // Size of each stick in bytes
    constexpr uint32_t aligned_stick_nbytes = get_compile_time_arg_val(4);   // Aligned size of each stick in bytes
    constexpr uint32_t tiles_per_channel_dim = get_compile_time_arg_val(5);  // Number of tiles per channel dimension
    constexpr uint32_t tiles_per_width_dim = get_compile_time_arg_val(6);    // Number of tiles per width dimension
    constexpr uint32_t element_size = get_compile_time_arg_val(7);           // Size of each element in bytes
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(8);            // Input circular buffer ID
    constexpr auto dst_args = TensorAccessorArgs<9>();

    // Runtime arguments - Processing parameters
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);        // Base destination address in DRAM
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);  // Starting block ID for processing
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);      // Number of blocks to process
    uint32_t patch_height_offset = get_arg_val<uint32_t>(3);   // Current height offset within patch
    uint32_t output_offset = get_arg_val<uint32_t>(4);         // Current output offset

    // Calculated constants
    constexpr uint32_t output_width = input_width / stride_width;  // Output tensor width
    constexpr uint32_t patch_size = stride_height * stride_width;  // Total elements per patch
    // Initialize DRAM address generator for interleaved memory access
    const auto dst = TensorAccessor(dst_args, dst_addr, stick_nbytes);

    // Processing loop bounds and state variables
    const uint32_t end_block_id = start_block_id + num_blocks;
    uint32_t curr_offset = output_offset;  // Current working offset
    uint32_t orig_patch_height_offset = patch_height_offset;

    // Main processing loop - iterate through each block
    for (uint32_t block_id = start_block_id; block_id < end_block_id; block_id++) {
        uint32_t stick_offset = 0;                // Current stick offset within patch
        uint32_t stride_width_idx = 0;            // Current position within stride width
        uint32_t output_stick_idx = curr_offset;  // Current destination stick index
        uint32_t remaining_width = input_width;   // Remaining width to process

        // Process each tile in the width dimension
        for (uint32_t tile_idx = 0; tile_idx < tiles_per_width_dim; tile_idx++) {
            cb_wait_front(input_cb_id, tiles_per_channel_dim);
            uint64_t l1_read_addr = get_write_ptr(input_cb_id);

            const uint32_t width_limit =
                (remaining_width < tt::constants::TILE_HEIGHT) ? remaining_width : tt::constants::TILE_HEIGHT;

            for (uint32_t stick_idx = 0; stick_idx < width_limit; stick_idx++) {
                const uint64_t dst_noc_addr = get_noc_addr(output_stick_idx, dst);
                noc_async_write(l1_read_addr, dst_noc_addr, stick_nbytes);

                // Update pointers and indices
                l1_read_addr += aligned_stick_nbytes;
                output_stick_idx++;
                stride_width_idx++;

                // Check if we've completed a stride width - move to next patch row
                if (stride_width_idx == stride_width) {
                    stick_offset++;
                    output_stick_idx = curr_offset + (stick_offset * patch_size);
                    stride_width_idx = 0;
                }
            }

            remaining_width -= tt::constants::TILE_HEIGHT;

            // Ensure all writes complete before moving to next set of tiles_per_channel_dim tiles
            noc_async_write_barrier();
            cb_pop_front(input_cb_id, tiles_per_channel_dim);
        }

        // Update patch offset for next block
        patch_height_offset++;

        // Check if we've completed a full patch height - move to next patch
        if (patch_height_offset == stride_height) {
            // Calculate new output offset for next patch
            curr_offset = output_offset + (patch_size * output_width) - (orig_patch_height_offset * stride_width);
            output_offset = curr_offset;
            patch_height_offset = 0;
            orig_patch_height_offset = 0;
        } else {
            // Move to next row within the same patch
            curr_offset += stride_width;
        }
    }
}
