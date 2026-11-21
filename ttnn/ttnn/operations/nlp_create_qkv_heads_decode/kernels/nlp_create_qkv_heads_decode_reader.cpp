// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// nlp_create_qkv_heads_decode — the `gather` kernel (op_design.md, regime A: direct gather).
//
// Runs on user u's core with `gather_threads` harts. The user's input row is row `user_row` of tile row
// `user_tile_row` of the interleaved input: `input_width_tiles` tiles, each holding the row as two face-row
// runs (left face, right face) of `face_row_bytes`. Section s (q heads, then k heads, then v heads) is
// tiles [s*head_dim_tiles, (s+1)*head_dim_tiles) of that tile row; head h of a tensor goes to row h of the
// tensor's shard (tile row h/32, row h%32), in the same tile column and face column.
//
// Every run is one NoC read from the input page straight into its final place in the local shard
// (LocalTensorAccessor destination, zero-copy): no ring, no staging, no writer. The runs are numbered flat
// over (section, head_dim tile, face column) and dealt round-robin: hart t issues runs t, t+R, ...
//
// Raw NoC calls, no kernel-library helper: no helper gathers sub-page runs of an interleaved tensor into a
// tensor shard (op_design.md "Helpers considered and rejected"); there is no compute stage.
//
// Every read carries the user transaction id (trid 0) and is drained by a barrier on trid 0: the kernel has
// no ring. The last statement hands the tag register back as 0 for the program that runs next.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;
constexpr uint32_t FACE_COLS = 2;  // face-row runs per tile row
constexpr NocOptVals user_txn = {.trid = 0};

struct GatherGeometry {
    uint32_t thread_id;          // this hart's place in the dealing
    uint32_t num_threads;        // the plan's R (the dealing stride)
    uint32_t first_page;         // input page of section 0, head_dim tile 0, of this user's tile row
    uint32_t src_row_offset;     // byte offset of this user's row in the left face run of an input tile
    uint32_t head_dim_tiles;
    uint32_t face_row_bytes;
    uint32_t face_bytes;         // 16 face rows: offset from a face to the next
    uint32_t out_page_bytes;     // the shard's tile stride
};

// gather_head_rows: issue this hart's share of the runs of sections [lo, hi) that belong to the tensor whose
// heads are sections [first_section, first_section + count), into that tensor's local shard `dst`.
// `phase` is the flat run index mod R, carried across calls so the dealing is global over the user's row.
template <typename Src, typename Dst>
FORCE_INLINE void gather_head_rows(
    const Noc& noc,
    const Src& src,
    const Dst& dst,
    const GatherGeometry& g,
    uint32_t first_section,
    uint32_t count,
    uint32_t lo,
    uint32_t hi,
    uint32_t& phase) {
    const uint32_t s_begin = lo > first_section ? lo : first_section;
    const uint32_t s_end = hi < first_section + count ? hi : first_section + count;
    for (uint32_t s = s_begin; s < s_end; ++s) {
        const uint32_t h = s - first_section;
        const uint32_t row_in_tile = h % TILE_ROWS;
        // Destination of head h, face column 0, head_dim tile 0: tile row h/32 of the shard, face row of h.
        const uint32_t dst_head = (h / TILE_ROWS) * g.head_dim_tiles * g.out_page_bytes +
                                  (row_in_tile / FACE_ROWS) * FACE_COLS * g.face_bytes +
                                  (row_in_tile % FACE_ROWS) * g.face_row_bytes;
        const uint32_t src_page = g.first_page + s * g.head_dim_tiles;
        for (uint32_t dt = 0; dt < g.head_dim_tiles; ++dt) {
            for (uint32_t c = 0; c < FACE_COLS; ++c) {
                if (phase == g.thread_id) {
                    noc.async_read<NocOptions::TXN_ID>(
                        src,
                        dst,
                        g.face_row_bytes,
                        {.page_id = src_page + dt, .offset_bytes = g.src_row_offset + c * g.face_bytes},
                        {.offset_bytes = dst_head + dt * g.out_page_bytes + c * g.face_bytes},
                        user_txn);
                }
                if (++phase == g.num_threads) {
                    phase = 0;
                }
            }
        }
    }
}

}  // namespace

void kernel_main() {
    // Thread plan: the plan's count, not get_num_threads(), so a hart that does not run leaves runs undone.
    constexpr uint32_t R = get_arg(args::gather_threads);
    constexpr uint32_t num_q_heads = get_arg(args::num_q_heads);
    constexpr uint32_t num_kv_heads = get_arg(args::num_kv_heads);
    constexpr uint32_t heads_per_block = get_arg(args::heads_per_block);
    constexpr uint32_t head_dim_tiles = get_arg(args::head_dim_tiles);
    constexpr uint32_t face_row_bytes = get_arg(args::face_row_bytes);
    constexpr uint32_t out_page_bytes = get_arg(args::out_page_bytes);
    constexpr uint32_t sections = num_q_heads + 2 * num_kv_heads;
    constexpr uint32_t input_width_tiles = sections * head_dim_tiles;  // the sections tile the input row (X4, X5)
    static_assert(heads_per_block >= 1, "heads_per_block must be at least 1");

    const uint32_t user_tile_row = get_arg(args::user_tile_row);
    const uint32_t user_row = get_arg(args::user_row);

    Noc noc;
    const auto in = TensorAccessor(tensor::in);
    const LocalTensorAccessor<uint32_t> q_shard(tensor::q);
    const LocalTensorAccessor<uint32_t> k_shard(tensor::k);
    const LocalTensorAccessor<uint32_t> v_shard(tensor::v);

    constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    const GatherGeometry g{
        .thread_id = get_my_thread_id(),
        .num_threads = R,
        .first_page = user_tile_row * input_width_tiles,
        .src_row_offset = (user_row / FACE_ROWS) * FACE_COLS * face_bytes + (user_row % FACE_ROWS) * face_row_bytes,
        .head_dim_tiles = head_dim_tiles,
        .face_row_bytes = face_row_bytes,
        .face_bytes = face_bytes,
        .out_page_bytes = out_page_bytes,
    };

    uint32_t phase = 0;
    for (uint32_t lo = 0; lo < sections; lo += heads_per_block) {
        const uint32_t hi = lo + heads_per_block < sections ? lo + heads_per_block : sections;
        gather_head_rows(noc, in, q_shard, g, 0, num_q_heads, lo, hi, phase);
        gather_head_rows(noc, in, k_shard, g, num_q_heads, num_kv_heads, lo, hi, phase);
        gather_head_rows(noc, in, v_shard, g, num_q_heads + num_kv_heads, num_kv_heads, lo, hi, phase);
        // drain_block: one barrier per block per hart.
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);
    }

    // release_tag: hand the hart's sticky tag register back as the user id.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
