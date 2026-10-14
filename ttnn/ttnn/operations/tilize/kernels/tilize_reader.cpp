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
    // Bank-coalesced stick reads (only when one column block spans the whole stick and images are
    // tile-row aligned): interleaved page p lives in bank p % num_banks at offset
    // (p / num_banks) * page, so the sticks of one bank inside a run of consecutive sticks are
    // contiguous there and come in ONE wide read into a reader-private staging buffer
    // (staging_tile_rows tile-rows per chunk, two slots so the next chunk's DRAM reads overlap
    // this chunk's local copies). Local L1->L1 NoC writes then restore row order for the tilize.
    constexpr bool coalesce_bank_reads = get_compile_time_arg_val(11) != 0;
    constexpr uint32_t num_banks = get_compile_time_arg_val(12);
    constexpr uint32_t cb_stick_staging = get_compile_time_arg_val(13);
    constexpr uint32_t staging_tile_rows = get_compile_time_arg_val(14);
    constexpr auto in_args = TensorAccessorArgs<15>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start_tile = get_arg_val<uint32_t>(3);
    const uint32_t block_height_tiles = get_arg_val<uint32_t>(4);
    const uint32_t bank_stagger_offset = get_arg_val<uint32_t>(5);  // < min(num_banks, chunk sticks)

    const auto in_acc = TensorAccessor(in_args, src_addr, stick_page_bytes);

    if constexpr (coalesce_bank_reads) {
        // Whole-stick single column block; tile-rows of this core are consecutive sticks.
        constexpr uint32_t chunk_sticks_max = staging_tile_rows * tile_h;
        constexpr uint32_t max_per_group = (chunk_sticks_max + num_banks - 1) / num_banks;
        constexpr uint32_t slot_bytes = num_banks * max_per_group * stick_page_bytes;
        constexpr uint32_t seg_bytes = block_width_tiles * tile_col_bytes;
        const uint32_t staging_base = get_write_ptr(cb_stick_staging);
        const uint32_t first_core_stick = row_start_tile * tile_h;  // images are tile-row aligned here

        auto issue_chunk = [&](uint32_t chunk_row0, uint32_t slot) {
            const uint32_t rows =
                (core_rows_tiles - chunk_row0 < staging_tile_rows) ? core_rows_tiles - chunk_row0 : staging_tile_rows;
            const uint32_t n_sticks = rows * tile_h;
            const uint32_t stick0 = first_core_stick + chunk_row0 * tile_h;
            const uint32_t base = staging_base + slot * slot_bytes;
            const uint32_t groups = n_sticks < num_banks ? n_sticks : num_banks;
            for (uint32_t k = 0; k < groups; ++k) {
                // Bank stagger: each core starts its bank walk at a different bank.
                uint32_t g = k + bank_stagger_offset;
                g = g >= groups ? g - groups : g;
                const uint32_t count = (n_sticks - g + num_banks - 1) / num_banks;
                noc_async_read(
                    in_acc.get_noc_addr(stick0 + g),
                    base + g * max_per_group * stick_page_bytes,
                    count * stick_page_bytes);
            }
        };

        // Local copies all target this core with the same size: program the write cmd buffer once
        // (the reader issues no other writes), then each copy only sets src/dst addresses.
        constexpr bool one_packet_copies = seg_bytes <= NOC_MAX_BURST_SIZE;
        if constexpr (one_packet_copies) {
            noc_async_write_one_packet_set_state(get_noc_addr(staging_base), seg_bytes);
        }

        uint32_t slot = 0;
        issue_chunk(0, 0);
        for (uint32_t chunk_row0 = 0; chunk_row0 < core_rows_tiles; chunk_row0 += staging_tile_rows) {
            noc_async_read_barrier();  // only this chunk's reads are outstanding
            const uint32_t next_row0 = chunk_row0 + staging_tile_rows;
            if (next_row0 < core_rows_tiles) {
                issue_chunk(next_row0, slot ^ 1);
            }
            const uint32_t base = staging_base + slot * slot_bytes;
            const uint32_t rows =
                (core_rows_tiles - chunk_row0 < staging_tile_rows) ? core_rows_tiles - chunk_row0 : staging_tile_rows;
            for (uint32_t t = 0; t < rows; ++t) {
                cb_reserve_back(cb_rm_sticks, block_width_tiles);
                uint32_t dst_l1 = get_write_ptr(cb_rm_sticks);
                for (uint32_t i = 0; i < tile_h; ++i) {
                    const uint32_t s = t * tile_h + i;  // stick index inside the chunk
                    const uint32_t src_l1 = base + ((s % num_banks) * max_per_group + s / num_banks) * stick_page_bytes;
                    if constexpr (one_packet_copies) {
                        noc_async_write_one_packet_with_state(src_l1, dst_l1);
                    } else {
                        noc_async_write(src_l1, get_noc_addr(dst_l1), seg_bytes);
                    }
                    dst_l1 += seg_bytes;
                }
                noc_async_write_barrier();
                cb_push_back(cb_rm_sticks, block_width_tiles);
            }
            slot ^= 1;
        }
        return;
    }

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
                for (uint32_t i = 0; i < tile_h; ++i) {
                    noc_async_read(in_acc.get_noc_addr(first_stick + i, seg_offset_bytes), l1_write_addr, seg_bytes);
                    l1_write_addr += seg_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_id, width_tiles);
            }
        }
    }
}
