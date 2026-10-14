// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader (NCRISC / NoC0) — block operation `load_sticks_block`.
//
// Walks this core's rectangle (row_start_tile, core_rows_tiles, col_start_tile, all columns)
// as column blocks of block_width_tiles (ragged last block = last_block_width_tiles, own CB ring),
// each split into row blocks of block_height_tiles. Per tile-row (the streaming quantum):
// reserve `width` tile-sized pages, issue tile_h stick-segment reads (all in flight), one
// barrier, push `width`. Page layout in the CB: tile_h rows of width*32 elements, which is
// what the symmetric tilize helper consumes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_rm_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_sticks_tail = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(3);  // 32 * elem bytes
    constexpr uint32_t image_height_sticks = get_compile_time_arg_val(4);
    constexpr uint32_t tile_rows_per_image = get_compile_time_arg_val(5);
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t last_block_width_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_col_blocks = get_compile_time_arg_val(9);
    constexpr bool has_tail = get_compile_time_arg_val(10) != 0;
    // Bank-coalesced stick reads (only when one column block spans the whole stick):
    // interleaved page p lives in bank p % num_banks at offset (p / num_banks) * page, so the
    // sticks of one bank inside a tile-row are contiguous there and come in ONE wide read into a
    // reader-private staging buffer; local L1 copies then restore row order for the tilize.
    constexpr bool coalesce_bank_reads = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t num_banks = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stick_staging = get_compile_time_arg_val(13);
    constexpr auto in_args = TensorAccessorArgs<14>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(3);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(4);

    const auto in_acc = TensorAccessor(in_args, src_addr, stick_page_bytes);

    for (uint32_t col_block = 0; col_block < num_col_blocks; ++col_block) {
        const bool is_tail = has_tail && (col_block == num_col_blocks - 1);
        const uint32_t cb_id = is_tail ? cb_rm_sticks_tail : cb_rm_sticks;
        const uint32_t width_tiles = is_tail ? last_block_width_tiles : block_width_tiles;
        const uint32_t seg_bytes = width_tiles * tile_col_bytes;
        const uint32_t seg_offset_bytes = (col_start_tile + col_block * block_width_tiles) * tile_col_bytes;

        for (uint32_t rb_start = 0; rb_start < core_rows_tiles; rb_start += block_height_tiles) {
            const uint32_t rb_end =
                (rb_start + block_height_tiles < core_rows_tiles) ? rb_start + block_height_tiles : core_rows_tiles;
            for (uint32_t r = rb_start; r < rb_end; ++r) {
                // Per-image ceil fold: global tile-row -> (image, tile-row within image).
                const uint32_t tile_row = row_start_tile + r;
                const uint32_t image = tile_row / tile_rows_per_image;
                const uint32_t row_in_image = tile_row - image * tile_rows_per_image;
                const uint32_t first_stick = image * image_height_sticks + row_in_image * tile_h;

                cb_reserve_back(cb_id, width_tiles);
                uint32_t l1_write_addr = get_write_ptr(cb_id);
                if constexpr (coalesce_bank_reads) {
                    // Groups: stick first_stick + g + m * num_banks, m = 0..count_g-1, all in one bank.
                    constexpr uint32_t num_groups = tile_h < num_banks ? tile_h : num_banks;
                    constexpr uint32_t max_per_group = (tile_h + num_banks - 1) / num_banks;
                    const uint32_t staging = get_write_ptr(cb_stick_staging);
                    for (uint32_t g = 0; g < num_groups; ++g) {
                        const uint32_t count = (tile_h - g + num_banks - 1) / num_banks;
#ifndef ABLATE_READS
                        noc_async_read(
                            in_acc.get_noc_addr(first_stick + g),
                            staging + g * max_per_group * stick_page_bytes,
                            count * stick_page_bytes);
#endif
                    }
                    noc_async_read_barrier();
                    for (uint32_t i = 0; i < tile_h; ++i) {
                        const uint32_t g = i % num_banks;
                        const uint32_t m = i / num_banks;
                        noc_async_read(
                            get_noc_addr(staging + (g * max_per_group + m) * stick_page_bytes),
                            l1_write_addr,
                            seg_bytes);
                        l1_write_addr += seg_bytes;
                    }
                } else {
                    for (uint32_t i = 0; i < tile_h; ++i) {
#ifndef ABLATE_READS
                        noc_async_read(
                            in_acc.get_noc_addr(first_stick + i, seg_offset_bytes), l1_write_addr, seg_bytes);
#endif
                        l1_write_addr += seg_bytes;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, width_tiles);
            }
        }
    }
}
