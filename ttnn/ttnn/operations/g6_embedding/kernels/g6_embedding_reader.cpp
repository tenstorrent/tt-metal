// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader: fetch_index_block + gather_rows_block.
//
// A block is (row_block, col_block): block_tile_rows tile-rows x block_col_tiles tile-columns.
// For each tile-row the reader issues one read per valid token: the table row named by the index,
// restricted to the block's column segment, landing at row r of the tile-row's row-major staging
// area. One read barrier per block retires both the row reads and the prefetch of the next block's
// indices, so index latency is off the critical path after the core's first block.
//
// No raw-LLK substitution: there is no compute here. Helpers considered (op_design.md, API
// Mapping): read_sticks_for_tilize reads contiguous page ids only, while the gather's page id is a
// data-dependent value loaded from L1.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t ELEMENT_BYTES = 2;
constexpr uint32_t INDEX_BYTES = 4;
}  // namespace

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    constexpr uint32_t index_depth = get_arg(args::index_depth);

    const uint32_t col_blocks_per_row = get_arg(args::col_blocks_per_row);
    const uint32_t tile_rows_per_image = get_arg(args::tile_rows_per_image);
    const uint32_t tensor_tile_rows = get_arg(args::tensor_tile_rows);
    const uint32_t token_extent = get_arg(args::token_extent);
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    // Bytes of one table-row segment: block_col_tiles whole tile-columns, so every segment offset and
    // length is a multiple of 64 B (Blackhole DRAM read alignment).
    constexpr uint32_t segment_bytes = block_col_tiles * TILE_DIM * ELEMENT_BYTES;
    constexpr uint32_t tile_row_bytes = TILE_DIM * segment_bytes;
    constexpr uint32_t tile_row_indices = TILE_DIM;
    constexpr uint32_t slot_indices = block_tile_rows * tile_row_indices;

    Noc noc;
    DataflowBuffer gathered_rows(dfb::gathered_rows);
    // volatile: the values land by NoC; loads must not be hoisted above the read barrier.
    Scratchpad<volatile uint32_t> index_values(scratch::index_values);
    const auto index_accessor = TensorAccessor(tensor::index);
    const auto table_accessor = TensorAccessor(tensor::table);

    auto tile_rows_in = [&](uint32_t row_block) -> uint32_t {
        const uint32_t remaining = tensor_tile_rows - row_block * block_tile_rows;
        return remaining < block_tile_rows ? remaining : block_tile_rows;
    };
    auto valid_rows_of = [&](uint32_t tile_row_in_image) -> uint32_t {
        const uint32_t remaining = token_extent - tile_row_in_image * TILE_DIM;
        return remaining < TILE_DIM ? remaining : TILE_DIM;
    };

    // Reads only the logical positions of each tile-row: a partial tile-row never reads past the
    // image's last token.
    auto fetch_index_block = [&](uint32_t row_block, uint32_t slot) {
        const uint32_t num_tile_rows = tile_rows_in(row_block);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            const uint32_t tile_row = row_block * block_tile_rows + t;
            const uint32_t image = tile_row / tile_rows_per_image;
            const uint32_t tile_row_in_image = tile_row - image * tile_rows_per_image;
            const uint32_t valid_rows = valid_rows_of(tile_row_in_image);
            noc.async_read(
                index_accessor,
                index_values,
                valid_rows * INDEX_BYTES,
                {.page_id = image, .offset_bytes = tile_row_in_image * tile_row_indices * INDEX_BYTES},
                {.offset_bytes = (slot * slot_indices + t * tile_row_indices) * INDEX_BYTES});
        }
    };

    uint32_t slot = 0;
    fetch_index_block(first_block / col_blocks_per_row, slot);
    noc.async_read_barrier();

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block = first_block + block_idx;
        const uint32_t row_block = block / col_blocks_per_row;
        const uint32_t col_block = block - row_block * col_blocks_per_row;
        const uint32_t num_tile_rows = tile_rows_in(row_block);
        const uint32_t block_pages = num_tile_rows * block_col_tiles;
        const uint32_t segment_offset = col_block * segment_bytes;

        gathered_rows.reserve_back(block_pages);
        for (uint32_t t = 0; t < num_tile_rows; ++t) {
            const uint32_t tile_row = row_block * block_tile_rows + t;
            const uint32_t tile_row_in_image = tile_row % tile_rows_per_image;
            const uint32_t valid_rows = valid_rows_of(tile_row_in_image);
            const uint32_t first_index = slot * slot_indices + t * tile_row_indices;
            // Rows >= valid_rows keep stale bytes: they become the unspecified padding of the tile.
            for (uint32_t r = 0; r < valid_rows; ++r) {
                const uint32_t table_row = index_values[first_index + r];
                noc.async_read<NocOptions::DEFAULT, segment_bytes>(
                    table_accessor,
                    gathered_rows,
                    segment_bytes,
                    {.page_id = table_row, .offset_bytes = segment_offset},
                    {.offset_bytes = t * tile_row_bytes + r * segment_bytes});
            }
        }

        // Every index of this block is now in registers or issued, so the next block's indices may
        // reuse a slot; never prefetch past the core's last block (no read left in flight at return).
        if (block_idx + 1 < num_blocks) {
            const uint32_t next_row_block = (block + 1) / col_blocks_per_row;
            if (next_row_block != row_block) {
                slot = slot + 1 == index_depth ? 0 : slot + 1;
                fetch_index_block(next_row_block, slot);
            }
        }

        noc.async_read_barrier();
        gathered_rows.push_back(block_pages);
    }
}
