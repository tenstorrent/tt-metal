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
//
// NATIVE SHARDED INPUT (`input_is_native`). When the block IS this core's own
// resident shard, cb_input_rows is PLACED ON the shard buffer by the host, so
// the block's bytes are already the CB's contents and there is nothing to move:
// `load_block` degenerates to marking the block's pages available. That is what
// consuming a shard means — an accessor read of a core's own shard would go out
// over the NoC to fetch bytes that are already in this core's L1. The accessor
// stays declared unconditionally (it owns the interleaved leg and the non-local
// cross-spec leg) so the compile-time arg indices never move.

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
    constexpr uint32_t input_is_native = get_compile_time_arg_val(7);
    constexpr uint32_t input_pages_per_row = get_compile_time_arg_val(8);
    constexpr uint32_t in_page_width_bytes = get_compile_time_arg_val(9);
    constexpr auto in_args = TensorAccessorArgs<10>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    // 1 on the solved plan (contiguous block ranges); the core count on the
    // shard-driven plan, where core i owns shards {i, i+N, i+2N, ...}.
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    // Stick-indexed accessor over the ROW_MAJOR input; page size comes from the
    // accessor's own compile-time args (the tensor's aligned stick size).
    [[maybe_unused]] const auto in_acc = TensorAccessor(in_args, src_addr);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        // resolve_block — index arithmetic only, from the same CT plan the
        // compute and writer kernels see.
        const uint32_t block_id = start_block_id + b * block_stride;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t w_chunk = block_id - row_group * num_w_chunks;

        // Balanced monotone row split; needs no remainder table.
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        if constexpr (!input_is_native && input_pages_per_row > 1) {
            // load_block, STRIDED. The source's shard cuts the width, so one row
            // spans `input_pages_per_row` pages and consecutive sticks are that
            // far apart in page index — `read_sticks_for_tilize` cannot express
            // it (it is stick-indexed by construction: `start_page + block_row +
            // row`, stride 1). RECORDED GAP: the helper would close this with a
            // `page_stride_per_row` parameter alongside `byte_offset_within_page`;
            // this branch is that parameter, written out. Everything else is the
            // helper's own shape — one reserve/read-burst/push per TILE-ROW, one
            // barrier per tile-row, `block_row_bytes` per stick.
            //
            // The host guarantees `block_row_bytes` divides `in_page_width_bytes`
            // (block_width_tiles is a common divisor of C and the page width in
            // tiles), so a block's row segment always sits inside ONE page.
            const uint32_t col_bytes = w_chunk * block_row_bytes;
            const uint32_t page_col = col_bytes / in_page_width_bytes;
            const uint32_t byte_in_page = col_bytes - page_col * in_page_width_bytes;
            for (uint32_t tr = 0; tr < block_row_extent; ++tr) {
                cb_reserve_back(cb_input_rows, block_width_tiles);
                uint32_t l1_write_addr = get_write_ptr(cb_input_rows);
                const uint32_t first_stick = (row_start + tr) * tile_h;
                for (uint32_t row = 0; row < tile_h; ++row) {
                    noc_async_read(
                        in_acc.get_noc_addr((first_stick + row) * input_pages_per_row + page_col, byte_in_page),
                        l1_write_addr,
                        block_row_bytes);
                    l1_write_addr += block_row_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_rows, block_width_tiles);
            }
        } else if constexpr (input_is_native) {
            // load_block, zero-copy: the block's `block_row_extent` tile-rows are
            // already resident in this core's L1 behind cb_input_rows. Marking
            // the whole block available in one push keeps the reader's quantum a
            // BLOCK, matching the accessor leg; the compute helper still waits
            // and pops one tile-row at a time.
            const uint32_t block_pages = block_row_extent * block_width_tiles;
            cb_reserve_back(cb_input_rows, block_pages);
            cb_push_back(cb_input_rows, block_pages);
        } else {
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
}
