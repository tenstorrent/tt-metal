// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// paged_update_cache: row_writer, the op's only kernel (regime direct_row_write, op_design.md).
//
// No kernel-library helper applies: the op computes nothing (no compute helper), and the data-movement
// helpers that exist address L1 -> L1 reads or tile columns staged in L1, while every transfer here is a
// face-row segment written from the resident input shard straight into a DRAM cache page at a byte offset
// (op_design.md, "Helpers considered and rejected"). Raw NoC calls are the block operations.
//
// Per core (one block: this core's users x heads x D-tiles x 2 face halves):
//   load_index_block  positions page + one page-table row per user of the block -> this thread's scratch slot,
//                     every read tagged trid 0, one read barrier
//   resolve_block     per user: p == -1 -> skip; else the cache tile-row base and the row inside the tile
//   write_rows_block  (h, j) units dealt round-robin over the plan's writer threads; each unit is two
//                     face_row_bytes segments, shard -> cache page + offset, tagged trid 0; one write barrier
//                     per writes_per_barrier
//   release_tags      tag registers back to the user id 0 for whatever program runs next
//
// Every transfer carries trid 0 (a user transaction id): an untagged transfer would reuse the hart's last tag,
// which after an implicit-sync program belongs to a ring.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {
constexpr int32_t SKIP_POSITION = -1;  // rule R4: this user writes nothing in this step
}  // namespace

void kernel_main() {
    const uint32_t first_user = get_arg(args::first_user);
    const uint32_t num_users = get_arg(args::num_users_this_core);

    // Tile geometry from the host (its single source; the host refuses tiles that are not 32x32 of 16x16 faces).
    constexpr uint32_t tile_rows = get_arg(args::tile_rows);
    constexpr uint32_t face_rows = get_arg(args::face_rows);
    constexpr uint32_t writer_threads = get_arg(args::writer_threads);
    constexpr uint32_t heads = get_arg(args::heads);
    constexpr uint32_t block_size_tiles = get_arg(args::block_size_tiles);
    constexpr uint32_t dtiles = get_arg(args::dtiles);
    constexpr uint32_t face_row_bytes = get_arg(args::face_row_bytes);
    constexpr uint32_t face_bytes = get_arg(args::face_bytes);
    constexpr uint32_t input_page_bytes = get_arg(args::input_page_bytes);
    constexpr uint32_t pos_page_bytes = get_arg(args::pos_page_bytes);
    constexpr uint32_t pt_row_bytes = get_arg(args::pt_row_bytes);
    constexpr uint32_t slot_bytes = get_arg(args::slot_bytes);
    constexpr uint32_t writes_per_barrier = get_arg(args::writes_per_barrier);
    constexpr NocOptVals user_txn = {.trid = 0};

    static_assert(face_row_bytes % NOC_DRAM_WRITE_ALIGNMENT_BYTES == 0, "face row must be a legal DRAM write size");
    static_assert(face_row_bytes % NOC_L1_READ_ALIGNMENT_BYTES == 0, "face row must be a legal L1 read size");
    static_assert(tile_rows == 2 * face_rows, "a tile row is two face rows");
    static_assert(pos_page_bytes % sizeof(int32_t) == 0 && pt_row_bytes % sizeof(int32_t) == 0);

    const uint32_t thread_id = get_my_thread_id();
    Noc noc;
    const auto positions = TensorAccessor(tensor::positions);
    const auto page_table = TensorAccessor(tensor::page_table);
    const auto cache = TensorAccessor(tensor::cache);
    const LocalTensorAccessor<uint32_t> input_shard(tensor::input);
    const Scratchpad<int32_t> index_scratch(scratch::index_scratch);

    // ---- load_index_block: this thread's slot = [positions page | page-table row of each user of the block]
    const uint32_t slot_offset = thread_id * slot_bytes;
    if (num_users > 0) {
        noc.async_read<NocOptions::TXN_ID>(
            positions, index_scratch, pos_page_bytes, {.page_id = 0}, {.offset_bytes = slot_offset}, user_txn);
        for (uint32_t i = 0; i < num_users; ++i) {
            noc.async_read<NocOptions::TXN_ID>(
                page_table,
                index_scratch,
                pt_row_bytes,
                {.page_id = first_user + i},
                {.offset_bytes = slot_offset + pos_page_bytes + i * pt_row_bytes},
                user_txn);
        }
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
    }

    uint32_t issued = 0;  // writes since the last barrier
    {
        // The NoC wrote the slot behind the DM core's cache: read it only under a lock of those bytes.
        constexpr uint32_t slot_words = slot_bytes / sizeof(int32_t);
        constexpr uint32_t pos_words = pos_page_bytes / sizeof(int32_t);
        constexpr uint32_t pt_row_words = pt_row_bytes / sizeof(int32_t);
        const uint32_t slot_word = slot_offset / sizeof(int32_t);
        auto lock = index_scratch.scoped_lock(slot_word, slot_words);

        for (uint32_t i = 0; i < num_users; ++i) {
            // ---- resolve_block (per user)
            const int32_t position = index_scratch[slot_word + first_user + i];
            if (position == SKIP_POSITION) {
                continue;
            }
            const uint32_t p = static_cast<uint32_t>(position);
            const uint32_t seq_tile_row = p / tile_rows;            // tile row of the position in the sequence
            const uint32_t k = seq_tile_row / block_size_tiles;        // its page-table entry (CT divisor)
            const uint32_t row_tile = seq_tile_row - k * block_size_tiles;  // tile row inside the block
            const uint32_t r = p % tile_rows;                        // row inside the tile
            const uint32_t block = static_cast<uint32_t>(index_scratch[slot_word + pos_words + i * pt_row_words + k]);
            // Destination offsets inside a cache tile: the two faces holding row r, at row r % 16.
            const uint32_t dst_row_offset = (2 * (r / face_rows)) * face_bytes + (r % face_rows) * face_row_bytes;

            // ---- write_rows_block (this user's units): unit (h, j) is this thread's when its turn comes round
            uint32_t turn = 0;
            for (uint32_t h = 0; h < heads; ++h) {
                const uint32_t tile_row_page = ((block * heads + h) * block_size_tiles + row_tile) * dtiles;
                const uint32_t src_row_offset =
                    (2 * (h / face_rows)) * face_bytes + (h % face_rows) * face_row_bytes;
                for (uint32_t j = 0; j < dtiles; ++j) {
                    const bool mine = (turn == thread_id);
                    if (++turn == writer_threads) {
                        turn = 0;
                    }
                    if (!mine) {
                        continue;
                    }
                    const uint32_t src_tile = (i * dtiles + j) * input_page_bytes;
                    for (uint32_t s = 0; s < 2; ++s) {  // the two face halves of a tile row
                        noc.async_write<NocOptions::TXN_ID>(
                            input_shard,
                            cache,
                            face_row_bytes,
                            {.offset_bytes = src_tile + src_row_offset + s * face_bytes},
                            {.page_id = tile_row_page + j, .offset_bytes = dst_row_offset + s * face_bytes},
                            user_txn);
                        if (++issued == writes_per_barrier) {
                            noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
                            issued = 0;
                        }
                    }
                }
            }
        }
    }
    if (issued > 0) {
        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);
    }

    // ---- release_tags: hand the hart back with the user id in both tag registers.
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc_async_write_set_trid(0, noc.get_noc_id());
}
