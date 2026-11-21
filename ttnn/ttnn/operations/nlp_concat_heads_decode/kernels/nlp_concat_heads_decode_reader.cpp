// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// nlp_concat_heads_decode gather kernel (regime `pull_rows`, op_design.md). Runs on output core `head_idx`.
//
// Helper substitution (design "Helpers considered and rejected"): raw Noc API, no kernel_lib helper. The op
// moves one sub-tile row of a remote tile into another row of a local tile: the compute copy helpers work on
// whole tiles in rings, `read_with_state` is local->local only, `zero_tile` zeroes one tile of a ring. Here the
// destination is the output tensor's own shard (LocalTensorAccessor, no ring) and the source a remote shard.
//
// Block schedule, per reader thread r of `reader_threads` (one block per thread):
//   zero_block:        tiles [t_begin, t_end) of the local output shard, one iDMA zero + one barrier
//   gather_rows_block: for every user u and tile t in [t_begin, t_end), the `runs_per_row` runs of row head_idx
//                      of input page u*head_tiles + t -> row u of local tile t; every read tagged trid 0,
//                      issued back to back, one barrier.
// Tile geometry: a row r of a tile is `runs_per_row` runs of `run_bytes`, run k in face
// (r / face_h) * runs_per_row + k, at byte (r % face_h) * run_bytes of that face.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {
// Byte offset of run k of row `row` inside a tile.
template <uint32_t face_bytes, uint32_t run_bytes, uint32_t face_h, uint32_t runs_per_row>
constexpr uint32_t row_run_offset(uint32_t row, uint32_t k) {
    return ((row / face_h) * runs_per_row + k) * face_bytes + (row % face_h) * run_bytes;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr uint32_t face_bytes = get_arg(args::face_bytes);
    constexpr uint32_t run_bytes = get_arg(args::run_bytes);
    constexpr uint32_t face_h = get_arg(args::face_h);
    constexpr uint32_t runs_per_row = get_arg(args::runs_per_row);
    constexpr uint32_t head_tiles = get_arg(args::head_tiles);
    constexpr uint32_t num_users = get_arg(args::num_users);
    constexpr uint32_t block_tiles = get_arg(args::block_tiles);
    constexpr uint32_t zero_padding_rows = get_arg(args::zero_padding_rows);
    constexpr uint32_t reader_threads = get_arg(args::reader_threads);
    // The host's thread split (one source: READER_THREADS -> block_tiles, reader_threads): the ranges cover the
    // head and no thread of the plan is empty.
    static_assert(block_tiles >= 1 && reader_threads >= 1, "empty thread plan");
    static_assert(reader_threads * block_tiles >= head_tiles, "thread ranges do not cover the head");
    static_assert((reader_threads - 1) * block_tiles < head_tiles, "a thread of the plan has no tile");

    const uint32_t head_idx = get_arg(args::head_idx);

    // This thread's contiguous tile range; the last one may be short, never empty (static_asserts above).
    const uint32_t t_begin = get_my_thread_id() * block_tiles;
    const uint32_t t_end = t_begin + block_tiles < head_tiles ? t_begin + block_tiles : head_tiles;

    Noc noc;
    const auto input = TensorAccessor(tensor::input);
    const LocalTensorAccessor<uint32_t> out_shard(tensor::output);

    if (t_begin < t_end) {
        // zero_block: rows users..31 are padding (R3); the reads below overwrite rows 0..users-1. The zero must
        // complete before any read lands in the same bytes.
        if constexpr (zero_padding_rows) {
            noc.async_write_zeros(out_shard, (t_end - t_begin) * tile_bytes, {.offset_bytes = t_begin * tile_bytes});
            noc.write_zeros_l1_barrier();
        }

        // gather_rows_block
        for (uint32_t u = 0; u < num_users; ++u) {
            for (uint32_t k = 0; k < runs_per_row; ++k) {
                const uint32_t src_off = row_run_offset<face_bytes, run_bytes, face_h, runs_per_row>(head_idx, k);
                const uint32_t dst_off = row_run_offset<face_bytes, run_bytes, face_h, runs_per_row>(u, k);
                for (uint32_t t = t_begin; t < t_end; ++t) {
                    noc.async_read<NocOptions::TXN_ID>(
                        input,
                        out_shard,
                        run_bytes,
                        {.page_id = u * head_tiles + t, .offset_bytes = src_off},
                        {.offset_bytes = t * tile_bytes + dst_off},
                        {.trid = 0});
                }
            }
        }
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 0});
    }

    // Hand the hart's sticky transaction tag back (target page, ring protocols): tt-metal programs that run next
    // issue untagged transfers.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
