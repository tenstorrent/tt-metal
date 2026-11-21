// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// embedding_gen reader: gather-direct (op_design.md).
//
// Raw data-movement API, no kernel-library helper: this op has no compute phase, and the DM helpers do not cover
// a gather (tilize/untilize helpers are compute-side and need whole tile-rows; mcast has no shared operand).
//
// Per block (token group x column chunk) of this core, thread t of R taking blocks t, t+R, ...:
//   prefetch the next own block's index slice into the other index slot; reserve one gathered_block entry;
//   issue every piece read of this block (no barrier inside); one read barrier; push the entry.
// Each piece lands at its final place in the output layout, so the writer stores the entry as is.
//
// Explicit sync on gathered_block (an entry is assembled from many reads): the host lists the ring in
// disable_dfb_implicit_sync_for, and every NoC transfer of this kernel carries user trid 0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "embedding_gen_common.hpp"

void kernel_main() {
    // Compile-time: the thread plan, the layouts, the tile geometry.
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr bool table_tiled = get_arg(args::table_tiled) != 0;
    constexpr bool out_tiled = get_arg(args::out_tiled) != 0;
    constexpr uint32_t index_slots = get_arg(args::index_slots);
    constexpr uint32_t block_tokens_max = get_arg(args::block_tokens_max);
    constexpr uint32_t index_bytes = get_arg(args::index_bytes);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t face_height = get_arg(args::face_height);
    constexpr uint32_t faces_per_row = get_arg(args::faces_per_row);  // 16-element halves of a tile row
    constexpr uint32_t col_unit_bytes = get_arg(args::col_unit_bytes);
    constexpr uint32_t face_row_bytes = get_arg(args::face_row_bytes);
    constexpr uint32_t face_bytes = get_arg(args::face_bytes);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr NocOptVals user_txn = {.trid = 0};

    // Per core: this core's contiguous range of units.
    const uint32_t start_unit = get_arg(args::start_unit);
    const uint32_t num_units = get_arg(args::num_units);
    // Common: the shape and the blocking (plan_blocks).
    const uint32_t V = get_arg(args::V);
    const embedding_gen::Geometry geo{
        get_arg(args::S),
        get_arg(args::Dt),
        get_arg(args::block_tokens),
        get_arg(args::block_cols),
        get_arg(args::num_col_chunks),
        get_arg(args::groups_per_batch)};
    const uint32_t Dt = geo.Dt;

    const uint32_t t = get_my_thread_id();
    DataflowBuffer ring(dfb::gathered_block);
    Scratchpad<uint32_t> token_indices(scratch::token_indices);
    Noc noc;
    const auto indices_acc = TensorAccessor(tensor::indices);
    const auto table_acc = TensorAccessor(tensor::table);

    // Row-major output: slab k of the entry is token k's `block_cols` column units.
    const uint32_t slab_bytes = geo.block_cols * col_unit_bytes;
    // This thread's index slots, in elements.
    const uint32_t slot_base = t * index_slots * block_tokens_max;

    // Face-row offset inside a 32x32 tile with four 16x16 faces: row r, half h (0: columns 0-15, 1: 16-31).
    auto face_off = [](uint32_t r, uint32_t h) -> uint32_t {
        return (r / face_height) * faces_per_row * face_bytes + h * face_bytes + (r % face_height) * face_row_bytes;
    };

    auto read_index_slice = [&](uint32_t local_block, uint32_t slot) {
        const embedding_gen::Block blk = geo.block(start_unit + local_block);
        noc.async_read<NocOptions::TXN_ID>(
            indices_acc,
            token_indices,
            blk.n_tok * index_bytes,
            {.page_id = blk.b, .offset_bytes = blk.s0 * index_bytes},
            {.offset_bytes = (slot_base + slot * block_tokens_max) * index_bytes},
            user_txn);
    };

    if (t < num_units) {
        read_index_slice(t, 0);  // load_index_slice: the only exposed index read
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
    }

    uint32_t ordinal = 0;  // this thread's block count: picks the index slot
    for (uint32_t local_block = t; local_block < num_units; local_block += R, ++ordinal) {
        const uint32_t slot = ordinal % index_slots;
        const embedding_gen::Block blk = geo.block(start_unit + local_block);

        // prefetch_index_slice: lands under this block's ring wait and gather, covered by its barrier. It is
        // issued before the reserve so a full ring (writer behind) hides it too; the slot it fills held the
        // previous block's indices, all of which that block's gather already loaded.
        if (local_block + R < num_units) {
            read_index_slice(local_block + R, (ordinal + 1) % index_slots);
        }

        ring.reserve_back(1);  // reserve_gathered_block

        // gather_block: every read of the block in flight together.
        const uint32_t slot_elems = slot_base + slot * block_tokens_max;
        for (uint32_t k = 0; k < blk.n_tok; ++k) {
            uint32_t v = token_indices[slot_elems + k];
            v = v < V ? v : V - 1;  // R2: an index beyond the table never forms an address outside it
            if constexpr (!table_tiled && !out_tiled) {
                // RM -> RM: the row's column chunk is one contiguous piece.
                noc.async_read<NocOptions::TXN_ID>(
                    table_acc,
                    ring,
                    blk.n_cols * col_unit_bytes,
                    {.page_id = v, .offset_bytes = blk.c0 * col_unit_bytes},
                    {.offset_bytes = k * slab_bytes},
                    user_txn);
            } else {
                // Face-row pieces: a column unit is faces_per_row 16-element halves, each contiguous on both sides.
                const uint32_t v_tile_row = v / tile_height;
                const uint32_t v_row = v % tile_height;
                for (uint32_t jj = 0; jj < blk.n_cols; ++jj) {
                    const uint32_t c = blk.c0 + jj;
                    for (uint32_t h = 0; h < faces_per_row; ++h) {
                        uint32_t src_page, src_off, dst_off;
                        if constexpr (table_tiled) {
                            src_page = v_tile_row * Dt + c;
                            src_off = face_off(v_row, h);
                        } else {
                            src_page = v;
                            src_off = c * col_unit_bytes + h * face_row_bytes;
                        }
                        if constexpr (out_tiled) {
                            dst_off = jj * tile_bytes + face_off(k, h);
                        } else {
                            dst_off = k * slab_bytes + jj * col_unit_bytes + h * face_row_bytes;
                        }
                        noc.async_read<NocOptions::TXN_ID>(
                            table_acc,
                            ring,
                            face_row_bytes,
                            {.page_id = src_page, .offset_bytes = src_off},
                            {.offset_bytes = dst_off},
                            user_txn);
                    }
                }
            }
        }

        // land_and_publish_block
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
        ring.push_back(1);
    }

    // Hand the hart's tag register back with the user id (target page, Ring protocols).
    noc_async_read_set_trid(0, noc.get_noc_id());
}
