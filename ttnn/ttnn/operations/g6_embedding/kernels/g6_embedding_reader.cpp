// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader: per block, gathers the table-row segments named by the block's index values
// into cb_gathered_rows (row-major, row r of tile-row t at (t*32 + r) * segment_bytes).
//
// Raw NoC reads instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper reads
// contiguous page ids (start_page + row); here each page id is a data-dependent index value loaded
// from L1. Its reserve -> reads -> one barrier -> push structure is kept. Closing the gap needs an
// index-driven variant taking a page-id source.
//
// cb_index_values is reader-private scratch (no CB credit): slots are ordered by this kernel's own
// read barriers. The next tile-row group's indices are prefetched after the current block's row
// reads are issued and before its barrier, so one barrier retires both.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {

constexpr uint32_t cb_index_values = get_compile_time_arg_val(0);
constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
constexpr uint32_t index_depth = get_compile_time_arg_val(4);
constexpr uint32_t col_blocks_per_row = get_compile_time_arg_val(5);
constexpr auto index_args = TensorAccessorArgs<6>();
constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

constexpr uint32_t tile_height = 32;
constexpr uint32_t tile_width = 32;
constexpr uint32_t index_row_bytes = tile_height * sizeof(uint32_t);
constexpr uint32_t index_slot_bytes = block_tile_rows * index_row_bytes;
constexpr uint32_t segment_bytes = block_col_tiles * tile_width * sizeof(uint16_t);
constexpr uint32_t tile_row_bytes = tile_height * segment_bytes;
constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

struct TokenGeometry {
    uint32_t tensor_token_extent;
    uint32_t tile_rows_per_image;
    uint32_t tensor_tile_rows;
};

// Rows of tile-row g that hold tokens; a block never mixes images within one tile-row.
struct TileRow {
    uint32_t image;
    uint32_t first_token;
    uint32_t valid_rows;
};

FORCE_INLINE TileRow tile_row_of(uint32_t g, const TokenGeometry& geom) {
    const uint32_t image = g / geom.tile_rows_per_image;
    const uint32_t first_token = (g - image * geom.tile_rows_per_image) * tile_height;
    const uint32_t remaining = geom.tensor_token_extent - first_token;
    return {image, first_token, remaining < tile_height ? remaining : tile_height};
}

// Issues the index reads of tile-row group `group` into `slot_addr`; no barrier of its own.
template <typename Accessor>
FORCE_INLINE void fetch_index_block(
    const Accessor& index_accessor, uint32_t group, uint32_t slot_addr, const TokenGeometry& geom) {
    for (uint32_t t = 0; t < block_tile_rows; ++t) {
        const uint32_t g = group * block_tile_rows + t;
        if (g >= geom.tensor_tile_rows) {
            break;
        }
        const TileRow row = tile_row_of(g, geom);
        noc_async_read(
            index_accessor.get_noc_addr(row.image, row.first_token * sizeof(uint32_t)),
            slot_addr + t * index_row_bytes,
            row.valid_rows * sizeof(uint32_t));
    }
}

// Issues one segment read per valid token of the group; rows past valid_rows keep stale bytes,
// which only reach the unspecified padding rows of the output.
template <typename Accessor>
FORCE_INLINE void gather_rows_block(
    const Accessor& weight_accessor,
    uint32_t group,
    uint32_t col_block,
    uint32_t slot_addr,
    uint32_t rows_addr,
    const TokenGeometry& geom) {
    const uint32_t segment_offset = col_block * segment_bytes;
    for (uint32_t t = 0; t < block_tile_rows; ++t) {
        const uint32_t g = group * block_tile_rows + t;
        if (g >= geom.tensor_tile_rows) {
            break;
        }
        const uint32_t valid_rows = tile_row_of(g, geom).valid_rows;
        volatile tt_l1_ptr uint32_t* indices = dataflow_kernel_lib::addr_to_l1_ptr(slot_addr + t * index_row_bytes);
        uint32_t dst_addr = rows_addr + t * tile_row_bytes;
        for (uint32_t r = 0; r < valid_rows; ++r) {
            noc_async_read(weight_accessor.get_noc_addr(indices[r], segment_offset), dst_addr, segment_bytes);
            dst_addr += segment_bytes;
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const TokenGeometry geom{
        get_common_arg_val<uint32_t>(2), get_common_arg_val<uint32_t>(3), get_common_arg_val<uint32_t>(4)};
    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);

    const uint32_t index_base = get_write_ptr(cb_index_values);
    uint32_t slot = 0;

    {
        DeviceZoneScopedN("fetch_index_block");
        fetch_index_block(index_accessor, first_block / col_blocks_per_row, index_base, geom);
        noc_async_read_barrier();
    }

    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        const uint32_t block_id = first_block + block_idx;
        const uint32_t group = block_id / col_blocks_per_row;
        const uint32_t col_block = block_id - group * col_blocks_per_row;
        const bool next_opens_group = block_idx + 1 < num_blocks_this_core && col_block + 1 == col_blocks_per_row;
        const uint32_t next_slot = slot + 1 == index_depth ? 0 : slot + 1;

        {
            DeviceZoneScopedN("rows_reserve");
            cb_reserve_back(cb_gathered_rows, block_pages);
        }
        {
            DeviceZoneScopedN("gather_rows_block");
            gather_rows_block(
                weight_accessor,
                group,
                col_block,
                index_base + slot * index_slot_bytes,
                get_write_ptr(cb_gathered_rows),
                geom);
            // Safe even when next_slot == slot (index_depth 1): every row read above already took its
            // address from the slot.
            if (next_opens_group) {
                fetch_index_block(index_accessor, group + 1, index_base + next_slot * index_slot_bytes, geom);
            }
            noc_async_read_barrier();
            cb_push_back(cb_gathered_rows, block_pages);
        }
        if (next_opens_group) {
            slot = next_slot;
        }
    }
}
