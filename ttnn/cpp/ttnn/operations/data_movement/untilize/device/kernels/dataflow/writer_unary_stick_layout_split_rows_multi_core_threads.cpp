// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-threaded variant of writer_unary_stick_layout_split_rows_multi_core.cpp: every thread waits for and
// pops every block (ALL consumer pattern), and thread t writes the rows j with j % num_threads == t. The
// per-row NoC commands are what bound an untilize writer, so they are spread over the DM harts.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // run-time args
    const auto num_input_blocks_to_process = get_arg(args::num_input_blocks_to_process);
    const auto height_wise_input_block_start_index = get_arg(args::height_wise_input_block_start_index);
    const auto num_unpadded_cols_per_input_block = get_arg(args::num_unpadded_cols_per_input_block);
    const auto width_wise_output_block_start_index = get_arg(args::width_wise_output_block_start_index);
    const auto num_cols_already_processed_in_first_output_block =
        get_arg(args::num_cols_already_processed_in_first_output_block);

    // compile-time args
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr auto num_output_blocks_across_width = get_arg(args::num_output_blocks_across_width);
    constexpr auto output_element_size = get_arg(args::output_element_size);
    constexpr auto num_cols_per_input_block = get_arg(args::num_cols_per_input_block);
    constexpr auto num_cols_per_output_block = get_arg(args::num_cols_per_output_block);

    const uint32_t my_thread = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    auto write_rows_in_current_block = [&](uint32_t block_height_index) {
        dfb_out.wait_front(num_tiles_per_input_block);

        uint32_t base_l1_read_addr = dfb_out.get_read_ptr();

        for (uint32_t j = my_thread; j < tile_height; j += num_threads) {
            uint32_t current_l1_read_addr = base_l1_read_addr + j * num_cols_per_input_block * output_element_size;

            uint32_t num_rows_already_processed = block_height_index * tile_height + j;
            uint32_t num_pages_already_processed_in_previous_rows =
                num_rows_already_processed * num_output_blocks_across_width;
            uint32_t output_page_id =
                num_pages_already_processed_in_previous_rows + width_wise_output_block_start_index;

            uint32_t num_cols_remaining_in_current_output_block =
                num_cols_per_output_block - num_cols_already_processed_in_first_output_block;
            uint32_t output_offset_within_page_in_bytes =
                num_cols_already_processed_in_first_output_block * output_element_size;

            uint32_t num_input_cols_processed = 0;
            while (num_input_cols_processed < num_unpadded_cols_per_input_block) {
                uint32_t num_cols_to_write = std::min(
                    num_unpadded_cols_per_input_block - num_input_cols_processed,
                    num_cols_remaining_in_current_output_block);
                uint32_t num_bytes_to_write = num_cols_to_write * output_element_size;

                CoreLocalMem<uint32_t> src(current_l1_read_addr);
                noc.async_write(
                    src,
                    s,
                    num_bytes_to_write,
                    {.offset_bytes = 0},
                    {.page_id = output_page_id, .offset_bytes = output_offset_within_page_in_bytes});

                num_input_cols_processed += num_cols_to_write;
                current_l1_read_addr += num_bytes_to_write;
                output_page_id++;
                num_cols_remaining_in_current_output_block = num_cols_per_output_block;
                output_offset_within_page_in_bytes = 0;
            }
        }

        noc.async_write_barrier();
        dfb_out.pop_front(num_tiles_per_input_block);
    };

    uint32_t height_wise_input_block_index = height_wise_input_block_start_index;
    for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
        write_rows_in_current_block(height_wise_input_block_index);
        height_wise_input_block_index++;
    }
}
