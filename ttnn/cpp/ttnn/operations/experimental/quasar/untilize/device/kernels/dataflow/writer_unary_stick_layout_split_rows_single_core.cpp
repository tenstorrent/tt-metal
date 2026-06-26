// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t num_blocks_across_height = get_arg(args::num_blocks_across_height);
    constexpr uint32_t num_output_columns_of_blocks = get_arg(args::num_output_columns_of_blocks);
    constexpr uint32_t num_blocks_per_output_column_row = get_arg(args::num_blocks_per_output_column_row);
    constexpr uint32_t num_tiles_per_output_block = get_arg(args::num_tiles_per_output_block);
    constexpr uint32_t output_single_block_width_size = get_arg(args::output_single_block_width_size);

    const auto s = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    uint32_t row_stick_ids[tile_height];
    uint32_t stick_offset = 0;

    auto write_tiles_in_current_block = [&]() {
        cb_out.wait_front(num_tiles_per_output_block);

        uint32_t l1_read_addr = cb_out.get_read_ptr();
        for (uint32_t l = 0; l < tile_height; ++l) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                output_single_block_width_size,
                {.offset_bytes = 0},
                {.page_id = row_stick_ids[l], .offset_bytes = stick_offset});
            l1_read_addr += output_single_block_width_size;
        }
        stick_offset += output_single_block_width_size;

        noc.async_write_barrier();
        cb_out.pop_front(num_tiles_per_output_block);
    };

    // Each row of tiles processed separately
    for (uint32_t i = 0; i < num_blocks_across_height; ++i) {
        for (uint32_t j = 0; j < num_output_columns_of_blocks; ++j) {
            // Determine the base stick_ids for the row of blocks in the current column
            for (uint32_t k = 0; k < tile_height; ++k) {
                uint32_t num_complete_rows_already_processed = (i * tile_height + k) * num_output_columns_of_blocks;
                row_stick_ids[k] = num_complete_rows_already_processed + j;
            }
            stick_offset = 0;

            for (uint32_t k = 0; k < num_blocks_per_output_column_row; ++k) {
                write_tiles_in_current_block();
            }
        }
    }
}
