// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (NCRISC / NoC0): fetch_index_block + gather_rows_block.
//
// A unit is (tile row, column piece): 32 token slots of one image x block_width_tiles tile columns.
// A block is block_height_tiles consecutive units of this core's run, in piece-major unit order.
// Per block: the table rows named by the block's index values are read as row slices into
// cb_row_staging, laid out row-major per unit (32 rows x row_slice_bytes) so that the tilize helper
// sees block_width_tiles symmetric tile pages per unit.
//
// Raw NoC calls instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper reads consecutive
// pages; the gather takes each row's page id from an index value read on the device, and the helper
// has no page-list parameter (missing block op: "read sticks for tilize from a page-id list").

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tile_constants.h"

namespace {

using tt::constants::TILE_HEIGHT;

// Walks units incrementally so the per-unit decode needs no division.
struct UnitCursor {
    uint32_t piece;
    uint32_t tile_row;
    uint32_t image;
    uint32_t tile_in_image;
};

template <uint32_t tensor_tile_rows, uint32_t tiles_per_image>
FORCE_INLINE UnitCursor cursor_at(uint32_t unit) {
    const uint32_t tile_row = unit % tensor_tile_rows;
    return UnitCursor{unit / tensor_tile_rows, tile_row, tile_row / tiles_per_image, tile_row % tiles_per_image};
}

template <uint32_t tensor_tile_rows, uint32_t tiles_per_image>
FORCE_INLINE void advance(UnitCursor& c) {
    if (++c.tile_in_image == tiles_per_image) {
        c.tile_in_image = 0;
        ++c.image;
    }
    if (++c.tile_row == tensor_tile_rows) {
        c.tile_row = 0;
        c.image = 0;
        ++c.piece;
    }
}

// Token slots of a unit that hold indices: the last tile row of each image may be partial, and its
// slots past the logical extent continue into the next image or the page padding.
template <uint32_t tiles_per_image, uint32_t last_tile_valid_rows>
FORCE_INLINE uint32_t valid_rows(const UnitCursor& c) {
    return c.tile_in_image + 1 == tiles_per_image ? last_tile_valid_rows : TILE_HEIGHT;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_row_staging = get_compile_time_arg_val(0);
    constexpr uint32_t cb_index_slots = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_height_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_image = get_compile_time_arg_val(5);
    constexpr uint32_t last_tile_valid_rows = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t index_slot_bytes = get_compile_time_arg_val(8);
    constexpr auto index_args = TensorAccessorArgs<9>();
    constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t row_slice_bytes = block_width_tiles * (tile_bytes / TILE_HEIGHT);
    constexpr uint32_t unit_bytes = TILE_HEIGHT * row_slice_bytes;
    constexpr uint32_t block_pages = block_height_tiles * block_width_tiles;
    static_assert(row_slice_bytes <= NOC_MAX_BURST_SIZE, "a row slice must be one NoC burst");

    const uint32_t unit_start = get_arg_val<uint32_t>(0);
    const uint32_t core_units = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);

    const auto index = TensorAccessor(index_args, get_common_arg_val<uint32_t>(0));
    const auto weight = TensorAccessor(weight_args, get_common_arg_val<uint32_t>(1));

    // cb_index_slots is reader-private scratch: written in place, never pushed or popped.
    const uint32_t index_slots = get_write_ptr(cb_index_slots);
    ASSERT((index_slots & 63) == 0);

    UnitCursor fetch_cursor = cursor_at<tensor_tile_rows, tiles_per_image>(unit_start);
    UnitCursor gather_cursor = fetch_cursor;

    // Index slices of n_units units into consecutive slots. Units of one image are adjacent slices of
    // one page, so each image's run is a single read.
    auto fetch_index_block = [&](uint32_t n_units) {
        uint32_t unit = 0;
        while (unit < n_units) {
            const uint32_t run_first = unit;
            const uint64_t src = index.get_noc_addr(fetch_cursor.image, fetch_cursor.tile_in_image * index_slot_bytes);
            uint32_t run_bytes;
            bool image_ends;
            do {
                run_bytes = (unit - run_first) * index_slot_bytes +
                            valid_rows<tiles_per_image, last_tile_valid_rows>(fetch_cursor) * sizeof(uint32_t);
                image_ends = fetch_cursor.tile_in_image + 1 == tiles_per_image;
                advance<tensor_tile_rows, tiles_per_image>(fetch_cursor);
                ++unit;
            } while (unit < n_units && !image_ends);
            noc_async_read(src, index_slots + run_first * index_slot_bytes, run_bytes);
        }
    };

    uint32_t units_left = core_units;
    auto units_in_block = [&]() { return units_left < block_height_tiles ? units_left : block_height_tiles; };

    {
        DeviceZoneScopedN("IDX-FETCH");
        fetch_index_block(units_in_block());
    }
    {
        DeviceZoneScopedN("IDX-WAIT");
        noc_async_read_barrier();
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t n_units = units_in_block();
        units_left -= n_units;

        uint32_t staging;
        {
            DeviceZoneScopedN("STAGE-RESERVE");
            cb_reserve_back(cb_row_staging, block_pages);
            staging = get_write_ptr(cb_row_staging);
            ASSERT((staging & 63) == 0);
        }
        {
            DeviceZoneScopedN("ROW-ISSUE");
            for (uint32_t unit = 0; unit < n_units; ++unit) {
                // The block's index values landed before the previous barrier, which invalidated the
                // L1 cache; every value is loaded before the next block's index read is issued.
                const volatile tt_l1_ptr uint32_t* ids =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_slots + unit * index_slot_bytes);
                const uint32_t column_offset = gather_cursor.piece * row_slice_bytes;
                const uint32_t rows = valid_rows<tiles_per_image, last_tile_valid_rows>(gather_cursor);
                uint32_t dst = staging + unit * unit_bytes;
                for (uint32_t slot = 0; slot < rows; ++slot) {
                    noc_async_read<NOC_MAX_BURST_SIZE>(
                        weight.get_noc_addr(ids[slot], column_offset), dst, row_slice_bytes);
                    dst += row_slice_bytes;
                }
                advance<tensor_tile_rows, tiles_per_image>(gather_cursor);
            }
        }
        if (units_left > 0) {
            // Stall shadow: the next block's index slices ride this block's row barrier.
            DeviceZoneScopedN("IDX-FETCH");
            fetch_index_block(units_in_block());
        }
        {
            // A ragged last block still pushes the nominal page count; its unread units are phantom
            // tile rows the writer never stores.
            DeviceZoneScopedN("ROW-WAIT");
            noc_async_read_barrier();
            cb_push_back(cb_row_staging, block_pages);
        }
    }
}
