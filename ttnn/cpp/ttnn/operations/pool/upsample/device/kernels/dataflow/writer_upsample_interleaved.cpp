// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    /*
    In the case the input was tiled, a single block refers to TILE_HEIGHT rows of data after untilization, block_height
    = TILE_HEIGHT

    In the case the input was ROW_MAJOR, a single block simply refers to a single output stick, in which case
    block_height = 1
    */
    uint32_t num_blocks_to_read = get_arg(args::num_blocks_to_read);
    uint32_t start_block_id = get_arg(args::start_block_id);

    constexpr auto output_page_size = get_arg(args::output_page_size);
    constexpr auto scale_h = get_arg(args::scale_h);
    constexpr auto scale_w = get_arg(args::scale_w);
    constexpr auto height = get_arg(args::height);
    constexpr auto width = get_arg(args::width);
    constexpr auto block_height = get_arg(args::block_height);
    constexpr auto num_tiles_per_block_row = get_arg(args::num_tiles_per_block_row);

    const auto s0 = TensorAccessor(tensor::output);

    DataflowBuffer out_dfb(dfb::out);
    Noc noc;

    constexpr uint32_t in_width = width / scale_w;
    constexpr uint32_t in_height = height / scale_h;
    uint32_t end_block_id = start_block_id + num_blocks_to_read;
    // reader copied the data from DRAM to CB buffer.
    // writer copy the data from CB buffer to DRAM.

    uint32_t current_stick = block_height * start_block_id;

    for (uint32_t b = start_block_id; b < end_block_id; b++) {
        out_dfb.wait_front(num_tiles_per_block_row);

        for (uint32_t in_block_row = 0; in_block_row < block_height; ++in_block_row) {
            uint32_t curr_index = current_stick % (in_width * in_height);
            uint32_t curr_batch = current_stick / (in_width * in_height);
            uint32_t x = curr_index / in_width;
            uint32_t y = curr_index % in_width;

            uint32_t read_offset = in_block_row * output_page_size;

            // calculate the start index where writer will start writing the data.
            // total --> scale_h * scale_w times data will be written to the DRAM.
            // offset calcutes the relative position of the data in the stick.
            uint32_t start_index = curr_batch * width * height + (scale_h * x) * width + scale_w * y;

            for (uint32_t j = 0; j < scale_h; j++) {
                for (uint32_t k = 0; k < scale_w; k++) {
                    uint32_t offset = j * width + k;

                    noc.async_write(
                        out_dfb,
                        s0,
                        output_page_size,
                        {.offset_bytes = read_offset},
                        {.page_id = start_index + offset});
                }
            }
            current_stick++;
        }
        noc.async_write_barrier();
        out_dfb.pop_front(num_tiles_per_block_row);
    }
}
