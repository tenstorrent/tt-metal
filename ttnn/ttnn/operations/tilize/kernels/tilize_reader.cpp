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
//
// PADDED INPUT (`pad_active`) — op_design.md's `grid2d_padded`, ADDITIVE on the
// block that already exists. The block grid, the core assignment, the CBs and
// the compute call are unchanged; only `load_block` gains a fill. Two pad
// regions exist and they are separate arithmetic:
//   * the W tail — bytes [valid_bytes, block_row_bytes) of a row that HAS data.
//     At most one tile's worth (32 elements), because C = ceil(W/32); filled in
//     place with `fill_l1_range`, which is alignment-aware and is exactly the
//     helper written for a row whose pad offset is not 4-byte aligned.
//   * a fully padded ROW — an H tail row, a row of an all-pad tile COLUMN
//     (`pad_mode="explicit"` past the tile round), or a row of an all-pad
//     leading-dim slice. Sourced by ONE local L1->L1 NoC read from `cb_pad_row`,
//     so the DM engine moves the bytes rather than the RISC storing them.
// The two phases are ordered read-then-fill with the barrier BETWEEN them, so a
// CPU store into the tail of a row can never race the NoC write into its head.
//
// The H tail is also why this branch cannot call `read_sticks_for_tilize`:
// that helper spans ONE contiguous stick run (`start_page + block_row + row`),
// which is only a valid tile-row index when `H % tile_h == 0` — with an H tail
// the source rows RESTART at every image boundary. So the block's read is
// SEGMENTED per image here, one reserve / read-burst / barrier / push per
// tile-row, which is the helper's own shape with the segmentation written out.
// RECORDED GAP: `rows_per_segment` + a `pad_value` pair (or a `fill` callback)
// alongside `byte_offset_within_page` would close both in the helper.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
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
    // --- padding. All eight are inert and the branch compiles out at pad_active == 0.
    constexpr uint32_t pad_active = get_compile_time_arg_val(10);
    constexpr uint32_t cb_pad_row = get_compile_time_arg_val(11);
    constexpr uint32_t elem_size = get_compile_time_arg_val(12);
    constexpr uint32_t pad_word = get_compile_time_arg_val(13);
    constexpr uint32_t in_num_images = get_compile_time_arg_val(14);
    constexpr uint32_t in_rows_per_image = get_compile_time_arg_val(15);   // the INPUT's logical H
    constexpr uint32_t in_row_bytes = get_compile_time_arg_val(16);        // the INPUT's logical W, in bytes
    constexpr uint32_t rows_per_image_out = get_compile_time_arg_val(17);  // tile-rows per image, padded
    constexpr auto in_args = TensorAccessorArgs<18>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block_id = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);
    // 1 on the solved plan (contiguous block ranges); the core count on the
    // shard-driven plan, where core i owns shards {i, i+N, i+2N, ...}.
    const uint32_t block_stride = get_arg_val<uint32_t>(3);

    // Stick-indexed accessor over the ROW_MAJOR input; page size comes from the
    // accessor's own compile-time args (the tensor's aligned stick size).
    [[maybe_unused]] const auto in_acc = TensorAccessor(in_args, src_addr);

    // Seed cb_pad_row: ONE block row of the fill, built once per kernel.
    // 32 elements go in by hand (<= 128 B, so the store loop is bounded by the
    // TILE WIDTH and not by the block width), then local L1->L1 reads DOUBLE
    // the filled span until the whole row is covered — log2(block_width_tiles)
    // transfers, the DM engine moving the bulk. Filling `block_row_bytes` with a
    // RISC store loop instead would be a per-word walk of up to 32 KB.
    [[maybe_unused]] uint32_t pad_row_addr = 0;
    if constexpr (pad_active) {
        pad_row_addr = get_write_ptr(cb_pad_row);
        constexpr uint32_t seed_bytes = block_row_bytes / block_width_tiles;  // == TILE_WIDTH * elem_size
        dataflow_kernel_lib::fill_l1_range<elem_size>(pad_row_addr, seed_bytes, pad_word);
        for (uint32_t filled = seed_bytes; filled < block_row_bytes;) {
            const uint32_t chunk = (filled < block_row_bytes - filled) ? filled : block_row_bytes - filled;
            noc_async_read(get_noc_addr(pad_row_addr), pad_row_addr + filled, chunk);
            noc_async_read_barrier();
            filled += chunk;
        }
    }

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

        if constexpr (pad_active) {
            // load_block, PADDED + SEGMENTED. `pad_active` implies the input is
            // read through the accessor (the host clears `input_native`, since a
            // resident shard holds no room for the fill).
            //
            // The column extent is the same for every tile-row of the block, so
            // the W-tail split is computed once here:
            //   valid_bytes = the part of this w_chunk that the input's logical
            //                 row actually reaches (0 for an all-pad column)
            //   pad_bytes   = the rest, < TILE_WIDTH * elem_size whenever
            //                 valid_bytes > 0 (C = ceil(W / TILE_WIDTH))
            const uint32_t col_bytes = w_chunk * block_row_bytes;
            const uint32_t valid_bytes =
                (in_row_bytes > col_bytes)
                    ? ((in_row_bytes - col_bytes < block_row_bytes) ? in_row_bytes - col_bytes : block_row_bytes)
                    : 0;
            const uint32_t pad_bytes = block_row_bytes - valid_bytes;
            // Only meaningful (and only used) when valid_bytes > 0, where the
            // host guarantees the whole segment sits inside ONE source page.
            const uint32_t page_col = col_bytes / in_page_width_bytes;
            const uint32_t byte_in_page = col_bytes - page_col * in_page_width_bytes;

            for (uint32_t tr = 0; tr < block_row_extent; ++tr) {
                // SEGMENTED per image: with an H tail the source sticks restart
                // at each image boundary, so the tile-row index has to be split
                // before it can become a stick index.
                const uint32_t global_tr = row_start + tr;
                const uint32_t image = global_tr / rows_per_image_out;
                const uint32_t first_src_row = (global_tr - image * rows_per_image_out) * tile_h;
                uint32_t valid_rows = 0;
                if (image < in_num_images && first_src_row < in_rows_per_image) {
                    valid_rows = in_rows_per_image - first_src_row;
                    if (valid_rows > tile_h) {
                        valid_rows = tile_h;
                    }
                }

                cb_reserve_back(cb_input_rows, block_width_tiles);
                const uint32_t block_addr = get_write_ptr(cb_input_rows);
                const uint32_t stick_base =
                    (image * in_rows_per_image + first_src_row) * input_pages_per_row + page_col;

                // Phase 1 — every NoC transfer of the tile-row, behind ONE barrier.
                uint32_t l1_write_addr = block_addr;
                for (uint32_t row = 0; row < tile_h; ++row) {
                    if (row < valid_rows && valid_bytes > 0) {
                        noc_async_read(
                            in_acc.get_noc_addr(stick_base + row * input_pages_per_row, byte_in_page),
                            l1_write_addr,
                            valid_bytes);
                    } else {
                        // A fully padded row: one L1 -> L1 transfer from the seeded row.
                        noc_async_read(get_noc_addr(pad_row_addr), l1_write_addr, block_row_bytes);
                    }
                    l1_write_addr += block_row_bytes;
                }
                noc_async_read_barrier();

                // Phase 2 — the W tail of the rows that carried data. AFTER the
                // barrier: these are RISC stores into the same L1 words the NoC
                // was just writing the head of, and a store issued while that
                // write is in flight is a race on the row's last aligned word.
                if (pad_bytes > 0 && valid_bytes > 0) {
                    l1_write_addr = block_addr + valid_bytes;
                    for (uint32_t row = 0; row < valid_rows; ++row) {
                        dataflow_kernel_lib::fill_l1_range<elem_size>(l1_write_addr, pad_bytes, pad_word);
                        l1_write_addr += block_row_bytes;
                    }
                }
                cb_push_back(cb_input_rows, block_width_tiles);
            }
        } else if constexpr (!input_is_native && input_pages_per_row > 1) {
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
