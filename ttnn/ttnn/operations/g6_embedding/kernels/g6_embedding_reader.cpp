// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (NCRISC, NoC0): index-driven row gather.
//
// Raw NoC reads, no helper: the gather reads table page idx[k] per token, a page id taken from L1
// at run time, which read_sticks_for_tilize (consecutive page ids) cannot express.
//
// Per block of up to block_tile_rows units (unit = tile row t x column piece p):
//   fetch_index_block(k+1)  index chunks of the next block into the other cb_index slot
//   gather_rows_block(k)    one row-slice read per valid token into cb_rows, one barrier, push
// cb_index is reader-private scratch used as a ring of index_depth slots (no push/pop).
// Only a core's last block may push fewer than block_tile_rows * col_block_tiles pages, so the
// cb_rows write pointer always lands exactly on its ring limit.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t TOKENS_PER_TILE_ROW = 32;
constexpr uint32_t INDEX_CHUNK_BYTES = TOKENS_PER_TILE_ROW * sizeof(uint32_t);
// One row of one tile column: 32 16-bit values.
constexpr uint32_t TILE_COLUMN_ROW_BYTES = 32 * sizeof(uint16_t);

// Walks units in column-piece-major order: unit = piece * tensor_tile_rows + tile_row, tile_row =
// image * image_tile_rows + image_tile_row. Divides once, then advances incrementally.
struct UnitCursor {
    uint32_t piece;
    uint32_t tile_row;
    uint32_t image;
    uint32_t image_tile_row;

    void init(uint32_t unit, uint32_t tensor_tile_rows, uint32_t image_tile_rows) {
        piece = unit / tensor_tile_rows;
        tile_row = unit - piece * tensor_tile_rows;
        image = tile_row / image_tile_rows;
        image_tile_row = tile_row - image * image_tile_rows;
    }

    void advance(uint32_t tensor_tile_rows, uint32_t image_tile_rows) {
        ++tile_row;
        ++image_tile_row;
        if (image_tile_row == image_tile_rows) {
            image_tile_row = 0;
            ++image;
        }
        if (tile_row == tensor_tile_rows) {
            tile_row = 0;
            image = 0;
            image_tile_row = 0;
            ++piece;
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t col_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t index_depth = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(5);
    constexpr uint32_t image_tile_rows = get_compile_time_arg_val(6);
    constexpr uint32_t tokens_per_image = get_compile_time_arg_val(7);
    constexpr uint32_t vocab_rows = get_compile_time_arg_val(8);
    constexpr auto index_args = TensorAccessorArgs<9>();
    constexpr auto table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t unit_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto table_accessor = TensorAccessor(table_args, table_addr);

    // Whole tile columns keep source offset and staging stride at 0 mod 64 (Blackhole DRAM rule).
    constexpr uint32_t slice_bytes = col_block_tiles * TILE_COLUMN_ROW_BYTES;
    constexpr uint32_t unit_rows_bytes = TOKENS_PER_TILE_ROW * slice_bytes;
    constexpr uint32_t slot_bytes = block_tile_rows * INDEX_CHUNK_BYTES;
    constexpr uint32_t last_table_row = vocab_rows - 1;

    const uint32_t num_blocks = (num_units + block_tile_rows - 1) / block_tile_rows;
    const uint32_t index_ring_base = get_write_ptr(cb_index);

    auto valid_rows = [](uint32_t image_tile_row) -> uint32_t {
        const uint32_t remaining = tokens_per_image - image_tile_row * TOKENS_PER_TILE_ROW;
        return remaining < TOKENS_PER_TILE_ROW ? remaining : TOKENS_PER_TILE_ROW;
    };
    auto units_in_block = [&](uint32_t block) -> uint32_t {
        const uint32_t left = num_units - block * block_tile_rows;
        return left < block_tile_rows ? left : block_tile_rows;
    };

    // Index chunks of one block: one read per maximal run of tile rows inside one image, since the
    // chunks of consecutive tile rows are contiguous bytes of that image's index page.
    UnitCursor fetch_cursor;
    fetch_cursor.init(unit_begin, tensor_tile_rows, image_tile_rows);
    auto fetch_index_block = [&](uint32_t block) {
        DeviceZoneScopedN("idx-fetch");
        const uint32_t slot_addr = index_ring_base + (block % index_depth) * slot_bytes;
        const uint32_t count = units_in_block(block);
        uint32_t j = 0;
        while (j < count) {
            const uint32_t run_image = fetch_cursor.image;
            const uint32_t run_first_row = fetch_cursor.image_tile_row;
            const uint32_t run_first_j = j;
            uint32_t last_valid = valid_rows(fetch_cursor.image_tile_row);
            fetch_cursor.advance(tensor_tile_rows, image_tile_rows);
            ++j;
            while (j < count && fetch_cursor.image == run_image &&
                   fetch_cursor.image_tile_row == run_first_row + (j - run_first_j)) {
                last_valid = valid_rows(fetch_cursor.image_tile_row);
                fetch_cursor.advance(tensor_tile_rows, image_tile_rows);
                ++j;
            }
            const uint32_t run_len = j - run_first_j;
            const uint32_t bytes = (run_len - 1) * INDEX_CHUNK_BYTES + last_valid * sizeof(uint32_t);
            noc_async_read(
                index_accessor.get_noc_addr(run_image, run_first_row * INDEX_CHUNK_BYTES),
                slot_addr + run_first_j * INDEX_CHUNK_BYTES,
                bytes);
        }
    };

    if (num_blocks == 0) {
        return;
    }

    // Block 0's indices gate every row read: issue them first, alone.
    fetch_index_block(0);
    noc_async_read_barrier();

    UnitCursor gather_cursor;
    gather_cursor.init(unit_begin, tensor_tile_rows, image_tile_rows);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // The prefetch retires at this block's barrier, before block + 1 reads its slot.
        if (block + 1 < num_blocks) {
            fetch_index_block(block + 1);
        }

        const uint32_t count = units_in_block(block);
        const uint32_t pages = count * col_block_tiles;
        {
            DeviceZoneScopedN("rows-reserve");
            cb_reserve_back(cb_rows, pages);
        }
        {
            DeviceZoneScopedN("rows-issue");
            const uint32_t slot_addr = index_ring_base + (block % index_depth) * slot_bytes;
            uint32_t rows_addr = get_write_ptr(cb_rows);
            for (uint32_t j = 0; j < count; ++j) {
                volatile tt_l1_ptr uint32_t* indices =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_addr + j * INDEX_CHUNK_BYTES);
                const uint32_t slice_offset = gather_cursor.piece * slice_bytes;
                const uint32_t rows = valid_rows(gather_cursor.image_tile_row);
                uint32_t dst = rows_addr;
                for (uint32_t r = 0; r < rows; ++r) {
                    uint32_t page = indices[r];
                    // Keeps an out-of-contract index inside the table allocation.
                    page = page < last_table_row ? page : last_table_row;
                    noc_async_read(table_accessor.get_noc_addr(page, slice_offset), dst, slice_bytes);
                    dst += slice_bytes;
                }
                rows_addr += unit_rows_bytes;
                gather_cursor.advance(tensor_tile_rows, image_tile_rows);
            }
        }
        {
            DeviceZoneScopedN("rows-wait");
            noc_async_read_barrier();
        }
        cb_push_back(cb_rows, pages);
    }
}
