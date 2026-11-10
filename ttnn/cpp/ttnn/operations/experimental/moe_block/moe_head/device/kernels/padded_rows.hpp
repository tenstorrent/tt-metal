// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A padded forward's real rows (routing_planner.hpp PADDED_CHUNK): PADDED_FORWARD_CHUNKS chunks packed in the forward,
// each SP row holding every chunk's rows_per_chunk rows back to back, chunk k's real rows a prefix of its block (main's
// rotated layout's closed form). The metadata at metadata_l1: the forward's metadata words (PADDED_FORWARD_METADATA: at
// PADDED_FORWARD_CHUNKS 1 (slot, start, end), above {num_chunks, 0} + PADDED_FORWARD_CHUNKS (slot, start, end)
// records), else one chunk's actual_start at metadata_l1 and actual_end at metadata_l1 + 64. SKIP_PADDING_CHUNKS's
// effective row chunks: a head's row chunks past the last with a real row are skipped (padded_live_chunks).
//
// The args come in as structs a kernel reads once (read_padded_forward_args / read_live_chunk_args from a pointer to
// its args in L1: moe_block_args::arg_pointer / common_pointer); nothing here reads an arg list itself.
#pragma once

#include <cstdint>
#include "api/tile_constants.h"

#ifndef PADDED_FORWARD_CHUNKS
#define PADDED_FORWARD_CHUNKS 1
#endif
static_assert(
    PADDED_FORWARD_CHUNKS >= 1 && PADDED_FORWARD_CHUNKS <= 8,
    "a forward of 1 .. 8 chunks (the metadata's 104 B in a 128 B slot)");

// Where a padded forward's metadata comes from and my SP row in it: five consecutive args, in this order
struct PaddedForwardArgs {
    uint32_t metadata_address;    // [0] actual_start's DRAM address, or the metadata words' (PADDED_FORWARD_METADATA)
    uint32_t actual_end_address;  // [1] actual_end's DRAM address (0 with PADDED_FORWARD_METADATA)
    uint32_t sp_row;              // [2] my SP row
    uint32_t sp_rows;             // [3] the SP rows
    uint32_t tokens_per_sp_row;   // [4] tokens an SP row holds
};
FORCE_INLINE PaddedForwardArgs read_padded_forward_args(const volatile tt_l1_ptr uint32_t* args) {
    return PaddedForwardArgs{args[0], args[1], args[2], args[3], args[4]};
}

// The live row chunks' args (SKIP_PADDING_CHUNKS, head_chip_program.hpp live_args): eleven consecutive args, in this
// order
struct LiveChunkArgs {
    PaddedForwardArgs forward;       // [0] .. [4]
    uint32_t first_row;              // [5] the call's first row in my SP row (a row block adds its stride)
    uint32_t row_chunks;             // [6] the chunks
    uint32_t chunk_row_tiles;        // [7] row tiles a chunk
    uint32_t owner_chunk_row_tiles;  // [8] global_row_tile's own chunk row tiles (0: chunk-major)
    uint32_t owner_row_tiles;        // [9] its own row tiles
    uint32_t metadata_scratch;       // [10] L1 scratch for the metadata (128 B from its first 64 B aligned address)
};
FORCE_INLINE LiveChunkArgs read_live_chunk_args(const volatile tt_l1_ptr uint32_t* args) {
    return LiveChunkArgs{read_padded_forward_args(args), args[5], args[6], args[7], args[8], args[9], args[10]};
}

// the metadata's slot in a scratch from address: its first 64 B aligned address and 64 B on (the DRAM source is 64 B
// aligned; into a 16 B aligned, not 64 B aligned slot the read came back zero) - give it 192 B
FORCE_INLINE uint32_t padded_metadata_slot(uint32_t address) { return (address + 63) & ~63u; }

// positions in [base, base + length) below end
FORCE_INLINE uint32_t padded_prefix_count(uint32_t end, uint32_t base, uint32_t length) {
    return end <= base ? 0 : (end - base < length ? end - base : length);
}
// SP row sp_row's real rows of a chunk [start, end) of rows_per_sp_row rows an SP row (moe_padding_config's closed
// form)
FORCE_INLINE uint32_t
padded_chunk_rows(uint32_t start, uint32_t end, uint32_t sp_row, uint32_t sp_rows, uint32_t rows_per_sp_row) {
    const uint32_t slab_rows = sp_rows * rows_per_sp_row, slab = start / slab_rows;
    const uint32_t boundary_sp_row = (start / rows_per_sp_row) % sp_rows, boundary_offset = start % rows_per_sp_row;
    if (sp_row != boundary_sp_row) {  // (one run: rows before the boundary row a slab on)
        const uint32_t run_slab = sp_row < boundary_sp_row ? slab + 1 : slab;
        return padded_prefix_count(end, run_slab * slab_rows + sp_row * rows_per_sp_row, rows_per_sp_row);
    }
    // (the boundary row starts mid-slab and wraps into the next: two runs)
    return padded_prefix_count(
               end, slab * slab_rows + sp_row * rows_per_sp_row + boundary_offset, rows_per_sp_row - boundary_offset) +
           padded_prefix_count(end, (slab + 1) * slab_rows + sp_row * rows_per_sp_row, boundary_offset);
}
// my SP row's real rows of each chunk of the forward (its block's real prefix), from the metadata words in L1
struct PaddedRows {
    uint32_t rows_per_chunk;                    // an SP row's rows a chunk
    uint32_t real_rows[PADDED_FORWARD_CHUNKS];  // chunk k's real rows: rows k rows_per_chunk .. + real_rows[k]
    // is row `row` of my SP row real (rows past every chunk's - the block's padding rows - are not)
    FORCE_INLINE bool real(uint32_t row) const {
        const uint32_t chunk = row / rows_per_chunk;
        return chunk < PADDED_FORWARD_CHUNKS && row - chunk * rows_per_chunk < real_rows[chunk];
    }
    // the real rows in [first_row, first_row + rows)
    FORCE_INLINE uint32_t count(uint32_t first_row, uint32_t rows) const {
        uint32_t real_count = 0;
        for (uint32_t chunk = 0; chunk < PADDED_FORWARD_CHUNKS; ++chunk) {
            const uint32_t chunk_first = chunk * rows_per_chunk, chunk_end = chunk_first + real_rows[chunk];
            const uint32_t overlap_first = chunk_first > first_row ? chunk_first : first_row;
            const uint32_t overlap_end = chunk_end < first_row + rows ? chunk_end : first_row + rows;
            real_count += overlap_end > overlap_first ? overlap_end - overlap_first : 0;
        }
        return real_count;
    }
};
FORCE_INLINE PaddedRows padded_rows_of(uint32_t metadata_l1, const PaddedForwardArgs& args) {
    PaddedRows rows;
    rows.rows_per_chunk = args.tokens_per_sp_row / PADDED_FORWARD_CHUNKS;
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_l1);
#ifdef PADDED_FORWARD_METADATA
    const uint32_t chunks_in_use = PADDED_FORWARD_CHUNKS == 1 ? 1u : words[0];  // (the rest are empty)
    for (uint32_t chunk = 0; chunk < PADDED_FORWARD_CHUNKS; ++chunk) {
        const uint32_t record = PADDED_FORWARD_CHUNKS == 1 ? 1u : 3u + 3u * chunk;
        rows.real_rows[chunk] =
            chunk < chunks_in_use
                ? padded_chunk_rows(words[record], words[record + 1], args.sp_row, args.sp_rows, rows.rows_per_chunk)
                : 0;
    }
#else
    rows.real_rows[0] = padded_chunk_rows(words[0], words[16], args.sp_row, args.sp_rows, rows.rows_per_chunk);
#endif
    return rows;
}

#ifndef COMPILE_FOR_TRISC
// The metadata read from DRAM into metadata_l1 (128 B, 64 B aligned: padded_metadata_slot), then my SP row's real rows.
// The scalars / words: one-page DRAM tensors (page 0 in bank 0).
FORCE_INLINE PaddedRows fetch_padded_rows(const PaddedForwardArgs& args, uint32_t metadata_l1) {
#ifdef PADDED_FORWARD_METADATA
    constexpr uint32_t bytes = PADDED_FORWARD_CHUNKS == 1 ? 16 : (4 * (2 + 3 * PADDED_FORWARD_CHUNKS) + 15) / 16 * 16;
    noc_async_read(get_noc_addr_from_bank_id<true>(0, args.metadata_address), metadata_l1, bytes);
#else
    noc_async_read(get_noc_addr_from_bank_id<true>(0, args.metadata_address), metadata_l1, 16);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, args.actual_end_address), metadata_l1 + 64, 16);
#endif
    noc_async_read_barrier();
    invalidate_l1_cache();
    return padded_rows_of(metadata_l1, args);
}
// global_row_tile (shared_expert_row_dataflow.cpp) from the args: chunk `chunk`'s chunk-local row tile
// local_row_tile's row tile of the call
FORCE_INLINE uint32_t padded_global_row_tile(
    uint32_t chunk,
    uint32_t local_row_tile,
    uint32_t chunk_row_tiles,
    uint32_t owner_chunk_row_tiles,
    uint32_t owner_row_tiles) {
    return owner_chunk_row_tiles ? (local_row_tile / owner_chunk_row_tiles) * owner_row_tiles +
                                       chunk * owner_chunk_row_tiles + local_row_tile % owner_chunk_row_tiles
                                 : chunk * chunk_row_tiles + local_row_tile;
}
// is row tile row_tile of the call (rows first_row + TILE_HEIGHT row_tile ..) real anywhere
FORCE_INLINE bool padded_tile_real(const PaddedRows& rows, uint32_t first_row, uint32_t row_tile) {
    return rows.count(first_row + tt::constants::TILE_HEIGHT * row_tile, tt::constants::TILE_HEIGHT) != 0;
}
// 1 + the last chunk with a real row tile (at least 1); first_row: the call's first row in my SP row (args.first_row,
// plus a row block's stride)
FORCE_INLINE uint32_t padded_live_chunks_of(const PaddedRows& rows, const LiveChunkArgs& args, uint32_t first_row) {
    for (uint32_t chunks = args.row_chunks; chunks > 1; --chunks) {
        for (uint32_t local_row_tile = 0; local_row_tile < args.chunk_row_tiles; ++local_row_tile) {
            if (padded_tile_real(
                    rows,
                    first_row,
                    padded_global_row_tile(
                        chunks - 1,
                        local_row_tile,
                        args.chunk_row_tiles,
                        args.owner_chunk_row_tiles,
                        args.owner_row_tiles))) {
                return chunks;
            }
        }
    }
    return 1;
}
// The live row chunks of a head call (SKIP_PADDING_CHUNKS): the metadata read into metadata_l1, then
// padded_live_chunks_of
FORCE_INLINE uint32_t padded_live_chunks(const LiveChunkArgs& args, uint32_t first_row, uint32_t metadata_l1) {
    return padded_live_chunks_of(fetch_padded_rows(args.forward, metadata_l1), args, first_row);
}
#endif
