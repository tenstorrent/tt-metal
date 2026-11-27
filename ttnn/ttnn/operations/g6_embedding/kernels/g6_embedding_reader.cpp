// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader of g6_embedding: fetch_index_block + gather_rows_block.
//
// Raw NoC reads instead of dataflow_kernel_lib::read_sticks_for_tilize: that helper reads contiguous
// sticks (page id = start + row), while the gather's page id is an index value loaded from L1. Its
// reserve -> reads -> one barrier -> push structure is kept per block.
//
// The index values never leave the device: each row block's indices are read into a reader-private
// slot of dfb::index_values and loaded by this RISC to address the table rows.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef G6_EMBEDDING_ZONES
#include "api/debug/kernel_profiler.hpp"
#define G6_ZONE(name) DeviceZoneScopedN(name)
#else
#define G6_ZONE(name)
#endif

void kernel_main() {
    constexpr uint32_t block_col_tiles = get_arg(args::block_col_tiles);
    constexpr uint32_t block_tile_rows = get_arg(args::block_tile_rows);
    constexpr uint32_t col_blocks_per_row = get_arg(args::col_blocks_per_row);
    constexpr uint32_t tensor_tile_rows = get_arg(args::tensor_tile_rows);
    constexpr uint32_t tile_rows_per_image = get_arg(args::tile_rows_per_image);
    constexpr uint32_t token_extent = get_arg(args::token_extent);
    constexpr uint32_t row_segment_bytes = get_arg(args::row_segment_bytes);
    constexpr uint32_t index_slot_bytes = get_arg(args::index_slot_bytes);

    constexpr uint32_t tile_height = 32;
    constexpr uint32_t index_bytes = sizeof(uint32_t);
    constexpr uint32_t tile_row_index_bytes = tile_height * index_bytes;
    constexpr uint32_t block_pages = block_tile_rows * block_col_tiles;
    // One tile-row of the staging block: 32 rows of one column segment = block_col_tiles tiles.
    constexpr uint32_t staging_tile_row_bytes = tile_height * row_segment_bytes;
    static_assert(index_slot_bytes == block_tile_rows * tile_row_index_bytes);

    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    DataflowBuffer dfb_index(dfb::index_values);
    DataflowBuffer dfb_rows(dfb::gathered_rows);
    const auto index_acc = TensorAccessor(tensor::index);
    const auto weight_acc = TensorAccessor(tensor::weight);

    // Only logical positions are read: a partial last tile-row reads valid_rows indices, never the
    // next image's page or page padding.
    auto valid_rows = [](uint32_t tile_row_in_image) -> uint32_t {
        const uint32_t remaining = token_extent - tile_row_in_image * tile_height;
        return remaining < tile_height ? remaining : tile_height;
    };

    // Issues the index reads of one row block into a fresh slot; retired by the caller's next barrier.
    auto fetch_index_block = [&](uint32_t row_block) {
        G6_ZONE("fetch_index_block");
        dfb_index.reserve_back(1);
        const uint32_t slot = dfb_index.get_write_ptr();
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t g = row_block * block_tile_rows + t;
            if (g >= tensor_tile_rows) {
                break;
            }
            const uint32_t image = g / tile_rows_per_image;
            const uint32_t tr = g - image * tile_rows_per_image;
            noc_async_read(
                index_acc.get_noc_addr(image, tr * tile_row_index_bytes),
                slot + t * tile_row_index_bytes,
                valid_rows(tr) * index_bytes);
        }
        dfb_index.push_back(1);
    };

    // Issues one read per valid token of the row block; rows past valid_rows keep stale bytes and
    // become the unspecified padding rows of the result.
    auto gather_rows_block = [&](uint32_t row_block, uint32_t col_block, uint32_t staging) {
        G6_ZONE("gather_rows_block");
        const auto* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_index.get_read_ptr());
        const uint32_t col_offset = col_block * row_segment_bytes;
        for (uint32_t t = 0; t < block_tile_rows; ++t) {
            const uint32_t g = row_block * block_tile_rows + t;
            if (g >= tensor_tile_rows) {
                break;
            }
            const uint32_t rows = valid_rows(g % tile_rows_per_image);
            const uint32_t tile_row_base = staging + t * staging_tile_row_bytes;
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(
                    weight_acc.get_noc_addr(indices[t * tile_height + r], col_offset),
                    tile_row_base + r * row_segment_bytes,
                    row_segment_bytes);
            }
        }
    };

    uint32_t row_block = first_block / col_blocks_per_row;
    uint32_t col_block = first_block - row_block * col_blocks_per_row;

    // The index read is the core's first transaction: every row read of the core depends on it.
    fetch_index_block(row_block);
    {
        G6_ZONE("index_barrier");
        noc_async_read_barrier();
    }
    dfb_index.wait_front(1);

    for (uint32_t b = 0; b < num_blocks; ++b) {
        {
            G6_ZONE("rows_reserve");
            dfb_rows.reserve_back(block_pages);
        }
        gather_rows_block(row_block, col_block, dfb_rows.get_write_ptr());

        const bool row_block_ends = col_block + 1 == col_blocks_per_row;
        const bool more_blocks = b + 1 < num_blocks;
        // Prefetch the next row block's indices behind this block's row reads so one barrier
        // retires both and the index latency leaves the critical path.
        if (row_block_ends && more_blocks) {
            fetch_index_block(row_block + 1);
        }
        {
            G6_ZONE("rows_barrier");
            noc_async_read_barrier();
        }
        dfb_rows.push_back(block_pages);

        if (row_block_ends) {
            dfb_index.pop_front(1);
            col_block = 0;
            ++row_block;
            if (more_blocks) {
                dfb_index.wait_front(1);
            }
        } else {
            ++col_block;
        }
    }
}
