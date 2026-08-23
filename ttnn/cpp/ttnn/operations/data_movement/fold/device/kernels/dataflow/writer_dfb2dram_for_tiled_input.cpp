// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "tt-metalium/constants.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <
    uint32_t input_width,            // Width of input tensor
    uint32_t stride_height,          // Vertical stride for fold
    uint32_t stride_width,           // Horizontal stride for fold
    uint32_t stick_nbytes,           // Size of each stick in bytes
    uint32_t aligned_stick_nbytes,   // Aligned size of each stick
    uint32_t tiles_per_channel_dim,  // Tiles per channel dimension
    uint32_t tiles_per_width_dim>    // Tiles per width dimension
TT_KERNEL void writer(
    // Runtime arguments - Processing parameters
    uint32_t start_block_id,       // Starting block ID for processing
    uint32_t num_blocks,           // Number of blocks to process
    uint32_t patch_height_offset,  // Current height offset within patch
    uint32_t output_offset) {      // Current output offset
    // Calculated constants
    constexpr uint32_t output_width = input_width / stride_width;  // Output tensor width
    constexpr uint32_t patch_size = stride_height * stride_width;  // Total elements per patch
    // Initialize DRAM address generator for interleaved memory access
    const auto dst = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_in1(dfb::in1);

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
            dfb_in1.wait_front(tiles_per_channel_dim);
            // Source the write pointer of the input DFB (matches the legacy WRITE_PTR selector).
            const uint32_t src_base = dfb_in1.get_write_ptr();
            uint32_t src_offset = 0;

            const uint32_t width_limit =
                (remaining_width < tt::constants::TILE_HEIGHT) ? remaining_width : tt::constants::TILE_HEIGHT;

            for (uint32_t stick_idx = 0; stick_idx < width_limit; stick_idx++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(src_base),
                    dst,
                    stick_nbytes,
                    {.offset_bytes = src_offset},
                    {.page_id = output_stick_idx});

                // Update pointers and indices
                src_offset += aligned_stick_nbytes;
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
            noc.async_write_barrier();
            dfb_in1.pop_front(tiles_per_channel_dim);
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
