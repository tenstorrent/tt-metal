// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    // compile-time args
    constexpr uint32_t dfb_id_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks_across_height = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_columns_of_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks_per_output_column_row = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_per_output_block = get_compile_time_arg_val(6);
    constexpr uint32_t output_single_block_width_size = get_compile_time_arg_val(7);

    constexpr auto dst_args = TensorAccessorArgs<8>();
    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out(dfb_id_out0);

    uint32_t row_stick_ids[tile_height];
    uint32_t stick_offset = 0;

    auto write_tiles_in_current_block = [&]() {
        dfb_out.wait_front(num_tiles_per_output_block);

        uint32_t l1_read_addr = dfb_out.get_read_ptr();
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
        dfb_out.pop_front(num_tiles_per_output_block);
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
