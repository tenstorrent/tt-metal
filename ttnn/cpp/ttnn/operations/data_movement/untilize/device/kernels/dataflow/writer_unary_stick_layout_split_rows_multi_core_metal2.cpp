// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_stick_layout_split_rows_multi_core.cpp.
//
// Bindings:
//   dfb::out                                            — DFB endpoint (CONSUMER)
//   ta::out                                             — TensorAccessor (output)
//   args::tile_height                                   — CTA
//   args::num_tiles_per_input_block                     — CTA
//   args::num_output_blocks_across_width                — CTA
//   args::output_element_size                           — CTA
//   args::num_cols_per_input_block                      — CTA
//   args::num_cols_per_output_block                     — CTA
//   args::num_input_blocks_to_process                   — RTA
//   args::height_wise_input_block_start_index           — RTA
//   args::num_unpadded_cols_per_input_block             — RTA
//   args::width_wise_output_block_start_index           — RTA
//   args::num_cols_already_processed_in_first_output_block — RTA
//
// Dead-CTA-slot note: legacy CTA[1] (output_stick_size) was unread by the kernel.
// Dropped in the metal2 fork.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_input_blocks_to_process = get_arg(args::num_input_blocks_to_process);
    auto height_wise_input_block_start_index = get_arg(args::height_wise_input_block_start_index);
    auto num_unpadded_cols_per_input_block = get_arg(args::num_unpadded_cols_per_input_block);
    auto width_wise_output_block_start_index = get_arg(args::width_wise_output_block_start_index);
    auto num_cols_already_processed_in_first_output_block =
        get_arg(args::num_cols_already_processed_in_first_output_block);

    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr auto num_output_blocks_across_width = get_arg(args::num_output_blocks_across_width);
    constexpr auto output_element_size = get_arg(args::output_element_size);
    constexpr auto num_cols_per_input_block = get_arg(args::num_cols_per_input_block);
    constexpr auto num_cols_per_output_block = get_arg(args::num_cols_per_output_block);

    const auto s = TensorAccessor(ta::out);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    auto write_tiles_in_current_block = [&](uint32_t block_height_index) {
        cb_out.wait_front(num_tiles_per_input_block);

        uint32_t base_l1_read_addr = cb_out.get_read_ptr();

        for (uint32_t j = 0; j < tile_height; ++j) {
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
        cb_out.pop_front(num_tiles_per_input_block);
    };

    uint32_t height_wise_input_block_index = height_wise_input_block_start_index;
    for (uint32_t i = 0; i < num_input_blocks_to_process; ++i) {
        write_tiles_in_current_block(height_wise_input_block_index);
        height_wise_input_block_index++;
    }
}
