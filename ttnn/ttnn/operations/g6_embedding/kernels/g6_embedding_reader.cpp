// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (NCRISC / NoC0): index-driven row gather.
//
// Per core: units [unit_begin, unit_begin + num_units), numbered column-piece-major
// (unit = p * tensor_tile_rows + t). A unit is one tile row (32 tokens of one image) by one
// column piece of col_block_tiles tile columns. Blocks are runs of block_tile_rows consecutive
// units; each unit carries its own (p, b, tr) decode, so a block may cross a column piece and only
// a core's last block is short, which keeps every cb_rows push but the last at the nominal page
// count (the ring pointer must land exactly on fifo_limit).
//
// Raw NoC reads instead of read_sticks_for_tilize: the row page id is an index value read from L1
// per token, which that helper's consecutive-page signature cannot carry.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t INDEX_BYTES = sizeof(uint32_t);
constexpr uint32_t INDEX_CHUNK_BYTES = TILE_ROWS * INDEX_BYTES;

// Walks the units of one core in order, advancing incrementally (one divide pair at start).
struct UnitCursor {
    uint32_t p;   // column piece
    uint32_t t;   // tile row of the tensor (b * image_tile_rows + tr)
    uint32_t b;   // image
    uint32_t tr;  // tile row inside the image

    template <uint32_t tensor_tile_rows, uint32_t image_tile_rows>
    static UnitCursor at(uint32_t unit) {
        UnitCursor c;
        c.p = unit / tensor_tile_rows;
        c.t = unit - c.p * tensor_tile_rows;
        c.b = c.t / image_tile_rows;
        c.tr = c.t - c.b * image_tile_rows;
        return c;
    }

    template <uint32_t tensor_tile_rows, uint32_t image_tile_rows>
    void advance() {
        ++t;
        ++tr;
        if (tr == image_tile_rows) {
            tr = 0;
            ++b;
        }
        if (t == tensor_tile_rows) {
            t = 0;
            b = 0;
            tr = 0;
            ++p;
        }
    }

    template <uint32_t image_tokens>
    uint32_t valid_rows() const {
        const uint32_t remaining = image_tokens - tr * TILE_ROWS;
        return remaining < TILE_ROWS ? remaining : TILE_ROWS;
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t col_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t index_depth = get_compile_time_arg_val(4);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(5);
    constexpr uint32_t image_tile_rows = get_compile_time_arg_val(6);
    constexpr uint32_t image_tokens = get_compile_time_arg_val(7);
    constexpr uint32_t table_rows = get_compile_time_arg_val(8);
    constexpr uint32_t index_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t table_page_bytes = get_compile_time_arg_val(10);
    constexpr auto index_args = TensorAccessorArgs<11>();
    constexpr auto table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    // Whole tile columns: offsets and strides stay multiples of 64 B (Blackhole DRAM read rule).
    constexpr uint32_t slice_bytes = col_block_tiles * TILE_ROWS * sizeof(uint16_t);
    constexpr uint32_t block_pages = block_tile_rows * col_block_tiles;
    constexpr uint32_t index_slot_bytes = block_tile_rows * INDEX_CHUNK_BYTES;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t unit_begin = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    const auto index_acc = TensorAccessor(index_args, index_addr, index_page_bytes);
    const auto table_acc = TensorAccessor(table_args, table_addr, table_page_bytes);

    const uint32_t num_blocks = (num_units + block_tile_rows - 1) / block_tile_rows;
    // cb_index is reader-private scratch: its slots form a ring addressed directly, never pushed.
    const uint32_t index_base = get_write_ptr(cb_index);

    auto units_in_block = [&](uint32_t block_idx) {
        const uint32_t done = block_idx * block_tile_rows;
        const uint32_t left = num_units - done;
        return left < block_tile_rows ? left : block_tile_rows;
    };
    auto index_slot = [&](uint32_t block_idx) { return index_base + (block_idx % index_depth) * index_slot_bytes; };

    // One read per run of tile rows that are contiguous in one index page; only valid positions.
    auto fetch_index_block = [&](uint32_t block_idx, UnitCursor& cursor) {
        DeviceZoneScopedN("idx-fetch");
        const uint32_t slot = index_slot(block_idx);
        const uint32_t units = units_in_block(block_idx);
        uint64_t run_src = 0;
        uint32_t run_dst = 0;
        uint32_t run_bytes = 0;
        uint32_t run_b = 0;
        uint32_t run_next_tr = 0;
        for (uint32_t j = 0; j < units; ++j) {
            const uint32_t chunk_bytes = cursor.valid_rows<image_tokens>() * INDEX_BYTES;
            if (run_bytes != 0 && cursor.b == run_b && cursor.tr == run_next_tr) {
                run_bytes += chunk_bytes;
            } else {
                if (run_bytes != 0) {
                    noc_async_read(run_src, run_dst, run_bytes);
                }
                run_src = index_acc.get_noc_addr(cursor.b, cursor.tr * INDEX_CHUNK_BYTES);
                run_dst = slot + j * INDEX_CHUNK_BYTES;
                run_bytes = chunk_bytes;
                run_b = cursor.b;
            }
            run_next_tr = cursor.tr + 1;
            cursor.advance<tensor_tile_rows, image_tile_rows>();
        }
        noc_async_read(run_src, run_dst, run_bytes);
    };

    // One row-slice read per valid token into its row of the block's cb_rows window.
    auto gather_rows_block = [&](uint32_t block_idx, UnitCursor& cursor, uint32_t rows_base) {
        DeviceZoneScopedN("rows-issue");
        const uint32_t units = units_in_block(block_idx);
        volatile tt_l1_ptr uint32_t* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_slot(block_idx));
        uint32_t dst = rows_base;
        for (uint32_t j = 0; j < units; ++j) {
            const uint32_t slice_offset = cursor.p * slice_bytes;
            const uint32_t valid = cursor.valid_rows<image_tokens>();
            for (uint32_t r = 0; r < valid; ++r) {
                uint32_t row = indices[r];
                // Out-of-contract values stay inside the table allocation.
                row = row < table_rows ? row : table_rows - 1;
                noc_async_read(table_acc.get_noc_addr(row, slice_offset), dst + r * slice_bytes, slice_bytes);
            }
            indices += TILE_ROWS;
            dst += TILE_ROWS * slice_bytes;
            cursor.advance<tensor_tile_rows, image_tile_rows>();
        }
    };

    UnitCursor fetch_cursor = UnitCursor::at<tensor_tile_rows, image_tile_rows>(unit_begin);
    UnitCursor gather_cursor = fetch_cursor;

    // The index read gates every row read, so it goes out before any bulk read.
    fetch_index_block(0, fetch_cursor);
    {
        DeviceZoneScopedN("rows-wait");
        noc_async_read_barrier();
    }

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Prefetch lands while this block waits for cb_rows space and for its row reads.
        if (block_idx + 1 < num_blocks) {
            fetch_index_block(block_idx + 1, fetch_cursor);
        }
        const uint32_t pages = units_in_block(block_idx) * col_block_tiles;
        {
            DeviceZoneScopedN("rows-reserve");
            cb_reserve_back(cb_rows, block_pages);
        }
        gather_rows_block(block_idx, gather_cursor, get_write_ptr(cb_rows));
        {
            DeviceZoneScopedN("rows-wait");
            noc_async_read_barrier();
        }
        cb_push_back(cb_rows, pages);
    }
}
