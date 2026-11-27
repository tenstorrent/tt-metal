// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding reader (NCRISC, NoC0): fetch_index_block + gather_rows_block.
//
// Raw NoC reads instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper reads
// consecutive pages, while the gather takes every row's page id from an index value read on the
// device. Missing block operation: "read sticks for tilize from a page-id list".
//
// Per block: one reserve, one segment read per valid token of each tile-row, then the next row
// group's index reads are issued before the block's single read barrier so they land under it.
// Index slots are only overwritten after the reads that consumed them have been issued, so any
// index_depth >= 1 is safe.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/scratchpad.h"
#include "api/tile_constants.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;
constexpr uint32_t COL_TILE_SEGMENT_BYTES = TILE_WIDTH * sizeof(uint16_t);  // one tile-column of a bf16 row
constexpr uint32_t INDEX_SLOT_BYTES = TILE_HEIGHT * sizeof(uint32_t);
}  // namespace

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t col_blocks_per_row = get_arg(args::col_blocks_per_row);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    constexpr uint32_t tensor_tile_rows = get_arg(args::tensor_tile_rows);
    constexpr uint32_t tile_rows_per_image = get_arg(args::tile_rows_per_image);
    constexpr uint32_t token_extent = get_arg(args::token_extent);
    constexpr uint32_t index_depth = get_arg(args::index_depth);
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    // Segment offsets and the L1 row pitch are whole tile-columns (multiples of 64 B), which keeps
    // L1 and DRAM addresses congruent mod 64 as Blackhole DRAM reads require.
    constexpr uint32_t segment_bytes = block_col_tiles * COL_TILE_SEGMENT_BYTES;
    constexpr uint32_t tile_row_bytes = TILE_HEIGHT * segment_bytes;

    Noc noc;
    DataflowBuffer gathered_rows(dfb::gathered_rows);
    Scratchpad<volatile uint32_t> index_values(scratch::index_values);
    volatile tt_l1_ptr uint32_t* index_words = &index_values.local_mem()[0];
    const auto index_accessor = TensorAccessor(tensor::index);
    const auto weight_accessor = TensorAccessor(tensor::weight);

    // Only the logical tokens of tile-row g are read: never the next image's page or page padding.
    auto valid_rows = [](uint32_t g) {
        const uint32_t tr = g % tile_rows_per_image;
        const uint32_t remaining = token_extent - tr * TILE_HEIGHT;
        return remaining < TILE_HEIGHT ? remaining : TILE_HEIGHT;
    };
    auto rows_in_group = [](uint32_t group) {
        const uint32_t remaining = tensor_tile_rows - group * block_tile_rows;
        return remaining < block_tile_rows ? remaining : block_tile_rows;
    };

    // fetch_index_block for every tile-row of a row group, into index slot ring position `slot`.
    auto fetch_index_group = [&](uint32_t group, uint32_t slot) {
        DeviceZoneScopedN("fetch_index_block");
        const uint32_t first_row = group * block_tile_rows;
        const uint32_t rows = rows_in_group(group);
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t g = first_row + j;
            const uint32_t image = g / tile_rows_per_image;
            const uint32_t tr = g - image * tile_rows_per_image;
            noc.async_read(
                index_accessor,
                index_values,
                valid_rows(g) * sizeof(uint32_t),
                {.page_id = image, .offset_bytes = tr * INDEX_SLOT_BYTES},
                {.offset_bytes = (slot * block_tile_rows + j) * INDEX_SLOT_BYTES});
        }
    };

    uint32_t group = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - group * col_blocks_per_row;
    uint32_t slot = 0;

    // The core's first transaction: nothing else can proceed without these indices.
    fetch_index_group(group, slot);
    noc.async_read_barrier();

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("gather_rows_block");
        const uint32_t rows = rows_in_group(group);
        const uint32_t pages = rows * block_col_tiles;
        const uint32_t col_offset = col_block * segment_bytes;

        gathered_rows.reserve_back(pages);
        for (uint32_t j = 0; j < rows; ++j) {
            const uint32_t g = group * block_tile_rows + j;
            const uint32_t valid = valid_rows(g);
            volatile tt_l1_ptr uint32_t* row_ids = index_words + (slot * block_tile_rows + j) * TILE_HEIGHT;
            const uint32_t tile_row_offset = j * tile_row_bytes;
            for (uint32_t r = 0; r < valid; ++r) {
                noc.async_read(
                    weight_accessor,
                    gathered_rows,
                    segment_bytes,
                    {.page_id = row_ids[r], .offset_bytes = col_offset},
                    {.offset_bytes = tile_row_offset + r * segment_bytes});
            }
        }

        if (++col_block == col_blocks_per_row) {
            col_block = 0;
            ++group;
            if (block_idx + 1 < num_blocks) {
                slot = (slot + 1) % index_depth;
                fetch_index_group(group, slot);
            }
        }

        noc.async_read_barrier();
        // Rows past `valid` keep stale bytes: they tilize into the output's unspecified padding.
        gathered_rows.push_back(pages);
    }
}
