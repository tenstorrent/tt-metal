// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// avg_pool2d reader (block operation `load_column_block`, op_design.md).
//
// Streams this cluster's channel tile columns [ct_start, ct_start + core_ct_tiles) of every position tile row
// into the `in` ring, block by block (block_ct_tiles columns per block), in the order the REDUCE_COL reduce
// helper consumes them: for each DEST chunk of the block (dest_chunk_tiles columns, the last one narrower),
// for each position tile row ht < Ht, for each column of the chunk. Page id = ht * tensor_ct_tiles + ct.
//
// The input may be interleaved (DRAM or L1) or width-sharded in L1: the accessor maps a page to its bank or
// shard core, one path for every placement.
//
// Rule R1 -- tile-padding rows. Rows [last_tile_valid_rows, 32) of every tile of the last position tile row
// are padding of undefined content. They are zeroed here, in L1, after the tile lands and before it is handed
// to compute.
//
// Ring protocol: explicit sync (the host opts `in` out of implicit sync), because an entry is assembled from
// a read and a zero fill. Batched: one reserve / one barrier / one push per chunk row (at most dest_chunk_tiles
// tiles), cut where the batch would straddle the end of the ring (a reserved span is written linearly from
// the write pointer). Every NoC transfer carries the user transaction id. write_zeros is an overlay iDMA
// transfer on Quasar, ordered after the read barrier and fenced by its own barrier before the push.
// Raw NoC API: no kernel-library helper drives an accessor-fed tile stream with a padding fill on Quasar.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tile_constants.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t ct_start = get_arg(args::ct_start);
    const uint32_t core_ct_tiles = get_arg(args::core_ct_tiles);
    const uint32_t block_ct_tiles = get_arg(args::block_ct_tiles);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t tensor_ct_tiles = get_arg(args::tensor_ct_tiles);
    const uint32_t last_tile_valid_rows = get_arg(args::last_tile_valid_rows);
    constexpr uint32_t dest_chunk_tiles = get_arg(args::dest_chunk_tiles);
    constexpr uint32_t in_ring_entries = get_arg(args::in_entries_per_neo);
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);
    // Thread plan (1, 1, 1) (op_design.md, Work Distribution). A plan with C > 1 deals whole DEST chunks to
    // NEOs (rounds padded); that dealing is not written yet.
    static_assert(R == 1 && C == 1, "avg_pool2d reader implements the (1, 1, 1) thread plan");
    static_assert(in_ring_entries >= dest_chunk_tiles, "a chunk row must fit in the input ring");

    DataflowBuffer in(dfb::in);
    Noc noc;
    const auto input = TensorAccessor(tensor::input);
    constexpr NocOptVals user_txn = {.trid = 0};
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc_async_write_set_trid(0, noc.get_noc_id());

    // Tile geometry (bfloat16 32x32: four 16x16 faces, row-major faces, row-major rows in a face).
    constexpr uint32_t tile_bytes = get_tile_size(dfb::in);
    constexpr uint32_t face_rows = tt::constants::FACE_HEIGHT;
    constexpr uint32_t num_faces_r = tt::constants::TILE_HEIGHT / tt::constants::FACE_HEIGHT;
    constexpr uint32_t num_faces_c = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
    constexpr uint32_t face_bytes = tile_bytes / (num_faces_r * num_faces_c);
    constexpr uint32_t row_bytes = face_bytes / face_rows;
    const bool mask_last_row = last_tile_valid_rows < face_rows * num_faces_r;

    // Zero rows [last_tile_valid_rows, 32) of the tile at byte offset tile_off of the reserved span.
    auto zero_padding_rows = [&](uint32_t tile_off) __attribute__((always_inline)) {
        for (uint32_t fr = 0; fr < num_faces_r; ++fr) {
            const uint32_t fr_row0 = fr * face_rows;
            const uint32_t first = last_tile_valid_rows > fr_row0 ? last_tile_valid_rows - fr_row0 : 0;
            if (first >= face_rows) {
                continue;
            }
            for (uint32_t fc = 0; fc < num_faces_c; ++fc) {
                const uint32_t face_off = tile_off + (fr * num_faces_c + fc) * face_bytes;
                noc.async_write_zeros(in, (face_rows - first) * row_bytes, {.offset_bytes = face_off + first * row_bytes});
            }
        }
    };

    uint32_t ring_pos = 0;  // entries pushed so far, modulo the ring: where the write pointer is
    for (uint32_t block_start = 0; block_start < core_ct_tiles; block_start += block_ct_tiles) {
        const uint32_t block_cols =
            (core_ct_tiles - block_start < block_ct_tiles) ? core_ct_tiles - block_start : block_ct_tiles;
        const uint32_t block_ct0 = ct_start + block_start;
        for (uint32_t chunk_start = 0; chunk_start < block_cols; chunk_start += dest_chunk_tiles) {
            const uint32_t chunk_cols =
                (block_cols - chunk_start < dest_chunk_tiles) ? block_cols - chunk_start : dest_chunk_tiles;
            const uint32_t chunk_ct0 = block_ct0 + chunk_start;
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                const bool zero_rows = mask_last_row && ht == Ht - 1;
                const uint32_t row_page0 = ht * tensor_ct_tiles + chunk_ct0;
                for (uint32_t j0 = 0; j0 < chunk_cols;) {
                    const uint32_t to_wrap = in_ring_entries - ring_pos;
                    const uint32_t n = (chunk_cols - j0 < to_wrap) ? chunk_cols - j0 : to_wrap;
                    in.reserve_back(n);
                    for (uint32_t j = 0; j < n; ++j) {
                        noc.async_read<NocOptions::TXN_ID>(
                            input, in, tile_bytes, {.page_id = row_page0 + j0 + j}, {.offset_bytes = j * tile_bytes},
                            user_txn);
                    }
                    noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
                    if (zero_rows) {
                        for (uint32_t j = 0; j < n; ++j) {
                            zero_padding_rows(j * tile_bytes);
                        }
                        noc.write_zeros_l1_barrier();
                    }
                    in.push_back(n);
                    ring_pos = (ring_pos + n == in_ring_entries) ? 0 : ring_pos + n;
                    j0 += n;
                }
            }
        }
    }
}
