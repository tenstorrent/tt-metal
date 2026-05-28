// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_stick_layout_split_rows_single_core.cpp.
//
// Bindings:
//   dfb::out                              — DFB endpoint (CONSUMER)
//   ta::out                               — TensorAccessor (output, interleaved)
//   args::tile_height                     — CTA
//   args::num_blocks_across_height        — CTA
//   args::num_output_columns_of_blocks    — CTA
//   args::num_blocks_per_output_column_row — CTA
//   args::num_tiles_per_output_block      — CTA
//   args::output_single_block_width_size  — CTA
//
// Dead-CTA-slot note: legacy slot 1 (output_stick_size) was unread by the kernel.
// Dropped in the metal2 fork.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto num_blocks_across_height = get_arg(args::num_blocks_across_height);
    constexpr auto num_output_columns_of_blocks = get_arg(args::num_output_columns_of_blocks);
    constexpr auto num_blocks_per_output_column_row = get_arg(args::num_blocks_per_output_column_row);
    constexpr auto num_tiles_per_output_block = get_arg(args::num_tiles_per_output_block);
    constexpr auto output_single_block_width_size = get_arg(args::output_single_block_width_size);

    const auto s = TensorAccessor(ta::out);

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

    for (uint32_t i = 0; i < num_blocks_across_height; ++i) {
        for (uint32_t j = 0; j < num_output_columns_of_blocks; ++j) {
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
