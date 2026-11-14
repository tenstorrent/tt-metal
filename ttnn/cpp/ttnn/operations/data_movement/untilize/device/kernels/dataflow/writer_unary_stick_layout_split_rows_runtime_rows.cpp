// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the interleaved multi-core untilize. Each stick is one full output page, at page id
// tile_row * tile_height + row-in-tile. This core's tile-row range comes from the common row count.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

void kernel_main() {
    // RUNTIME ARGS
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);

    // COMPILE TIME ARGS
    constexpr auto num_cores = get_arg(args::num_cores);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto num_tiles_per_input_block = get_arg(args::num_tiles_per_input_block);
    constexpr auto num_output_blocks_across_width = get_arg(args::num_output_blocks_across_width);
    constexpr auto output_element_size = get_arg(args::output_element_size);
    constexpr auto num_cols_per_input_block = get_arg(args::num_cols_per_input_block);
    constexpr auto num_cols_per_output_block = get_arg(args::num_cols_per_output_block);

    // Interleaved input and output: one input block and one output page per row, both the tensor width.
    static_assert(num_output_blocks_across_width == 1, "untilize writer: interleaved output has one page per row");
    static_assert(
        num_cols_per_output_block == num_cols_per_input_block, "untilize writer: a stick is one full output page");
    static_assert(num_cols_per_input_block > 0, "untilize writer: the row must hold a column");
    constexpr uint32_t stick_bytes = num_cols_per_input_block * output_element_size;

    const auto [row_start, row_end] = ttnn::runtime_length::chunked_range(core_idx, num_tile_rows, num_cores);

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t block = row_start; block < row_end; ++block) {
        dfb_out.wait_front(num_tiles_per_input_block);
        const uint32_t base_l1_read_addr = dfb_out.get_read_ptr();
        for (uint32_t j = 0; j < tile_height; ++j) {
            CoreLocalMem<uint32_t> src(base_l1_read_addr + j * stick_bytes);
            noc.async_write(src, s, stick_bytes, {.offset_bytes = 0}, {.page_id = block * tile_height + j});
        }
        noc.async_write_barrier();
        dfb_out.pop_front(num_tiles_per_input_block);
    }
}
