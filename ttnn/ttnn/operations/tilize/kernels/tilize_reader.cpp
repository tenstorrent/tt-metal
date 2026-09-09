// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NoC0) — `load_block`.
//
// One `read_sticks_for_tilize` call per BLOCK. The helper owns the whole block:
// it reserves `block_width_tiles` pages, issues `tile_h` NoC reads behind ONE
// barrier, and pushes — once per tile-row, for all `block_row_extent` tile-rows
// of the block. Nothing here is per-tile-row.
//
// The three block extents map onto the helper's parameters exactly:
//   total_num_rows          = block_row_extent * tile_h   (the tile_row extent)
//   row_bytes               = block_row_bytes             (the tile_col extent)
//   byte_offset_within_page = w_chunk * block_row_bytes    (the column-chunk id)
//
// `byte_offset_within_page` is the helper's documented wide-W chunking
// parameter: it selects this block's column slice INSIDE each stick page, so
// the CB footprint scales with the chunk width and not with the tensor's W.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);  // R
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t block_row_bytes = get_compile_time_arg_val(6);  // block_width_tiles*32*elem
    constexpr auto in_args = TensorAccessorArgs<7>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);

    // Stick-indexed accessor over the ROW_MAJOR input; page size comes from the
    // accessor's own compile-time args (the tensor's aligned stick size).
    const auto in_acc = TensorAccessor(in_args, src_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        // resolve_block — index arithmetic only, from the same CT plan the
        // compute and writer kernels see.
        const uint32_t block_id = start_block_id + b;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        // Balanced monotone row split; needs no remainder table.
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        // load_block. Valid as one contiguous stick run because H % tile_h == 0
        // on the tile-aligned path, so tile-row r starts at stick r * tile_h
        // exactly, even where R comes from the leading-dim fold.
        dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rows, dataflow_kernel_lib::TilizeGranularity::TILE>(
            in_acc,
            /* total_num_rows          */ block_row_extent * tile_h,
            /* row_bytes               */ block_row_bytes,
            /* start_page              */ row_start * tile_h,
            /* byte_offset_within_page */ w_chunk * block_row_bytes);
    }
}
