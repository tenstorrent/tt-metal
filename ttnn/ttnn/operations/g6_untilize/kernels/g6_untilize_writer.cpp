// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// store_row_block (BRISC, NoC1): each gathered row of cb_row_block leaves as one ROW_MAJOR page chunk.
// The helper waits/pops block_tile_cols tile pages per <= tile_h-row block and write-barriers each
// block, so no write is in flight when the kernel returns.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "g6_untilize_work_split.hpp"

void kernel_main() {
    constexpr uint32_t cb_row_block = get_compile_time_arg_val(0);
    constexpr uint32_t tensor_rows = get_compile_time_arg_val(1);
    constexpr uint32_t num_cores = get_compile_time_arg_val(2);
    constexpr uint32_t rect_w = get_compile_time_arg_val(3);
    constexpr uint32_t row_elems = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(5);
    constexpr uint32_t block_tile_cols = get_compile_time_arg_val(6);
    constexpr auto output_args = TensorAccessorArgs<7>();

    constexpr uint32_t tile_w = unpack_tile_c_dim[cb_row_block];
    constexpr uint32_t elem_bytes = get_tile_size(cb_row_block) / get_tile_hw(cb_row_block);
    constexpr uint32_t num_col_blocks = tensor_tile_cols / block_tile_cols;
    constexpr uint32_t block_elems = block_tile_cols * tile_w;

    const g6_untilize::CoreRows mine = g6_untilize::core_rows<tensor_rows, num_cores, rect_w>();
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
        const uint32_t col_elem0 = col_block_idx * block_elems;
        // The last chunk may end inside a partial tile; it still spans block_tile_cols CB pages.
        const uint32_t chunk_elems = (row_elems - col_elem0 < block_elems) ? row_elems - col_elem0 : block_elems;
        {
            // Zone-only split: the helper's own wait then returns at once.
            DeviceZoneScopedN("G6U-store-wait");
            cb_wait_front(cb_row_block, block_tile_cols);
        }
        {
            DeviceZoneScopedN("G6U-store");
            dataflow_kernel_lib::write_sticks_after_untilize<cb_row_block>(
                output, mine.count, chunk_elems * elem_bytes, mine.start, col_elem0 * elem_bytes);
        }
    }
}
