// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (NCRISC / NoC0): fetch_index_block + gather_rows_block.
//
// A block is (row group gb of block_tile_rows tile-rows, column block c of block_col_tiles tile
// columns). For each tile-row g of the block it reads the valid_rows(g) uint32 indices of that
// tile-row into a private slot of cb_index_values, then one table-row segment per index into row r
// of the block's row-major staging area in cb_gathered_rows. Index values are read by this RISC
// from L1 and never leave the device.
//
// No kernel_lib helper is used for the two operations: read_sticks_for_tilize addresses contiguous
// page ids only, while the gather's page id is a value loaded from L1.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tile_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace {

using tt::constants::TILE_HEIGHT;
constexpr uint32_t INDEX_SLOT_BYTES = TILE_HEIGHT * sizeof(uint32_t);

struct TileRow {
    uint32_t image;
    uint32_t tile_row_in_image;
    uint32_t valid_rows;
};

FORCE_INLINE TileRow locate(uint32_t g, uint32_t token_extent, uint32_t tile_rows_per_image) {
    const uint32_t image = g / tile_rows_per_image;
    const uint32_t tr = g - image * tile_rows_per_image;
    const uint32_t remaining = token_extent - tr * TILE_HEIGHT;
    return {image, tr, remaining < TILE_HEIGHT ? remaining : TILE_HEIGHT};
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_index_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gathered_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block_col_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_tile_rows = get_compile_time_arg_val(3);
    constexpr uint32_t index_depth = get_compile_time_arg_val(4);
    constexpr auto index_args = TensorAccessorArgs<5>();
    constexpr auto weight_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;

    const uint32_t index_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t weight_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t token_extent = get_common_arg_val<uint32_t>(2);
    const uint32_t tile_rows_per_image = get_common_arg_val<uint32_t>(3);
    const uint32_t tensor_tile_rows = get_common_arg_val<uint32_t>(4);
    const uint32_t col_blocks_per_row = get_common_arg_val<uint32_t>(5);

    const uint32_t first_block = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);

    const auto index_accessor = TensorAccessor(index_args, index_addr);
    const auto weight_accessor = TensorAccessor(weight_args, weight_addr);

    // One tile column of one row-major row: TILE_WIDTH 16-bit values.
    const uint32_t row_bytes_per_tile_col = get_tile_size(cb_gathered_rows) / TILE_HEIGHT;
    const uint32_t segment_bytes = block_col_tiles * row_bytes_per_tile_col;
    const uint32_t index_base = get_write_ptr(cb_index_values);

    auto fetch_index_block = [&](uint32_t row_group, uint32_t slot) {
        DeviceZoneScopedN("fetch_index_block");
        const uint32_t slot_addr = index_base + slot * block_tile_rows * INDEX_SLOT_BYTES;
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t g = row_group * block_tile_rows + t;
            if (g >= tensor_tile_rows) {
                break;
            }
            const TileRow row = locate(g, token_extent, tile_rows_per_image);
            noc_async_read(
                index_accessor.get_noc_addr(row.image, row.tile_row_in_image * INDEX_SLOT_BYTES),
                slot_addr + t * INDEX_SLOT_BYTES,
                row.valid_rows * sizeof(uint32_t));
        }
    };

    auto gather_rows_block = [&](uint32_t row_group, uint32_t col_block, uint32_t slot, uint32_t l1_block) {
        DeviceZoneScopedN("gather_rows_block");
        const uint32_t slot_addr = index_base + slot * block_tile_rows * INDEX_SLOT_BYTES;
        const uint32_t segment_offset = col_block * segment_bytes;
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t g = row_group * block_tile_rows + t;
            if (g >= tensor_tile_rows) {
                break;
            }
            const TileRow row = locate(g, token_extent, tile_rows_per_image);
            volatile tt_l1_ptr uint32_t* indices =
                dataflow_kernel_lib::addr_to_l1_ptr(slot_addr + t * INDEX_SLOT_BYTES);
            uint32_t l1_row = l1_block + t * TILE_HEIGHT * segment_bytes;
            for (uint32_t r = 0; r < row.valid_rows; ++r) {
                noc_async_read(weight_accessor.get_noc_addr(indices[r], segment_offset), l1_row, segment_bytes);
                l1_row += segment_bytes;
            }
        }
    };

    uint32_t row_group = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_group * col_blocks_per_row;
    uint32_t slot = 0;

    // The core's first transaction: its first block cannot issue a row read before these land.
    fetch_index_block(row_group, slot);
    noc_async_read_barrier();

    for (uint32_t block = 0; block < num_blocks; ++block) {
        cb_reserve_back(cb_gathered_rows, block_pages);
        gather_rows_block(row_group, col_block, slot, get_write_ptr(cb_gathered_rows));

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++row_group;
            // Prefetch the next row group's indices behind this block's row reads so the barrier
            // below retires both; never past the core's last block (BB-03).
            if (block + 1 < num_blocks) {
                slot = (slot + 1 == index_depth) ? 0 : slot + 1;
                fetch_index_block(row_group, slot);
            }
        }

        {
            DeviceZoneScopedN("gather_rows_barrier");
            noc_async_read_barrier();
        }
        // Nominal page count even for a partial tile-row or a short last row group: the
        // producer and consumer quanta stay equal so the ring closes.
        cb_push_back(cb_gathered_rows, block_pages);
    }
}
