// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified topk_large_indices compute kernel. The factory selects the role per
// CreateKernel site via compile-time defines:
//
//   (no role define) -- ROW-PARALLEL: each core reduces full rows and
//   materializes the index output itself. The body mode is picked by a second
//   define set from shape: FUSED_E2E (stay fused through every merge/rebuild,
//   one global split per row), FUSED_SEGMENTED (per-<=32-chunk-segment fusion
//   folded into an unfused cross-segment survivor), or neither (classic
//   unfused per-chunk split, with the data-dependent chunk-skip early-out).
//
//   TOPK_TREE -- COLUMN-PARALLEL tree node (every rectangle core except the
//   root). Phase 1 (leaf): reduce this core's chunk slice to one K survivor in
//   DST. Phase 2 (tree): for each of this core's num_merges winning levels,
//   consume one partner sequence from the recv CB (delivered by this core's
//   writer via the pairwise handshake), copy it to DST operand-1, and
//   merge+rebuild IN PLACE -- the survivor never leaves DST between levels.
//   Phase 3 (ship): pack the survivor raw (FP32 values + UINT32 indices) for
//   the writer to send to this core's winner.
//
//   TOPK_TREE + TOPK_TREE_ROOT -- COLUMN-PARALLEL tree root (rectangle core
//   (0,0), slice 0). Same leaf reduction and in-DST tree merges, but the root
//   never ships: every rebuild stays DESCENDING (it is always the next merge's
//   operand 0), and after the last level it materializes the indices exactly
//   like the row-parallel epilogue.
//
// Tree directions: the bitonic halver needs operand0 descending and operand1
// ascending, so every DST-mutating action rebuilds DESCENDING except a node's
// LAST one before shipping, which rebuilds ASCENDING (validated upstream by
// test_topk_xl_rebuild_ascending: rebuild(asc) after a merge is the exact
// rank-mirror of rebuild(desc)).
//
// Tree empty slice (valid_length cut, nodes only -- slice 0 always holds at
// least one element):
//   * num_merges == 0 -> nothing to compute; the writer ships the prefilled
//     all--inf scratch sequence instead.
//   * num_merges > 0 -> the FIRST partner is ADOPTED (copied to operand-0 and
//     rebuilt) instead of merged; an ascending monotonic run is bitonic, so
//     the rebuild is valid in either direction.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

#include "topk_large_indices_compute_common.hpp"

#ifdef TOPK_TREE
#include "api/compute/tile_move_copy.h"  // copy_tile of received sequences
#ifndef TOPK_TREE_ROOT
#include "api/compute/pack.h"  // pack_block of the shipped survivor
#endif
#endif
#if !defined(TOPK_TREE) || defined(TOPK_TREE_ROOT)
#include "api/compute/pack_untilize.h"  // rank-order index materialization
#endif

#ifndef TOPK_TREE

#include "topk_large_indices_chunk_skip.hpp"

// Data-dependent chunk-skip early-out (row-parallel path only; see
// topk_large_indices_chunk_skip.hpp for design + soundness proof).
// One-line A/B toggle: flip to false to reproduce the pre-skip kernel
// (kernels JIT-compile, no host rebuild needed for the toggle).
constexpr bool kChunkSkipEnable = true;

// Bring-up diagnostic: dump the post-rebuild DST values region over DPRINT to
// calibrate rank_to_values_word against torch rank order. Never enable
// together with performance runs.
// #define CHUNK_SKIP_DIAG 1
#ifdef CHUNK_SKIP_DIAG
#include "api/debug/dprint.h"
#endif

namespace {

using topk_large_indices::copy_chunk;
using topk_large_indices::finish_chunk_classic;

// Second half of chunk processing: stamp fused LSB indices and locally sort.
// FUSED_E2E rows stamp the RUNTIME chunk id into index bits [15:11] and stay
// fused through every merge/rebuild (one global split per row at the end); the
// classic path splits to unfused values + row-major indices per chunk and
// advances the chunk base.
template <uint32_t K>
FORCE_INLINE void finish_chunk(uint32_t dst_base, bool ascending, uint32_t chunk_id) {
#ifdef FUSED_E2E
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst_base, chunk_id);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst_base, ascending);
#else
    (void)chunk_id;
    finish_chunk_classic<K>(dst_base, ascending);
#endif
}

#ifdef FUSED_SEGMENTED
// Segmented fusion (rows wider than 32 chunks): each <=32-chunk segment runs
// the FUSED_E2E flow with a segment-LOCAL 5-bit chunk id, is split once with
// the segment base OR'd into the decoded indices, and is folded into the
// running cross-segment survivor with one unfused merge. See the LLK notes on
// _topk_xl_separate_indices_row_major_global_base_.
template <uint32_t K>
FORCE_INLINE void finish_chunk_fused_local(uint32_t dst_base, bool ascending, uint32_t local_id) {
    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices_rt<K>(dst_base, local_id);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(dst_base, ascending);
}
#endif

template <uint32_t K>
FORCE_INLINE void process_chunk(
    CircularBuffer& input_cb, uint32_t dst_base, uint32_t active_elements, bool ascending, uint32_t chunk_id) {
    copy_chunk<K>(input_cb, dst_base, active_elements);
    finish_chunk<K>(dst_base, ascending, chunk_id);
}

}  // namespace

#endif  // !TOPK_TREE

void kernel_main() {
    using namespace topk_large_indices;

    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_chunks = get_arg_val<uint32_t>(1);
    const uint32_t tail_elements = get_arg_val<uint32_t>(2);

#if defined(TOPK_TREE_ROOT)
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t recv_cb = get_compile_time_arg_val(2);
    constexpr uint32_t K = get_compile_time_arg_val(3);
#elif defined(TOPK_TREE)
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t ship_values_cb = get_compile_time_arg_val(1);
    constexpr uint32_t ship_indices_cb = get_compile_time_arg_val(2);
    constexpr uint32_t recv_cb = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
#else
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t indices_cb = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t USER_K = get_compile_time_arg_val(3);
#endif

    static_assert(K == 512 || K == 1024 || K == 2048, "K must be 512, 1024, or 2048");
    constexpr uint32_t tiles_per_sequence = (K + elements_per_tile - 1) / elements_per_tile;
    constexpr uint32_t sequence_tiles = tiles_per_sequence * 2u;
    constexpr uint32_t slot0 = 0;

#ifdef TOPK_TREE
    // ------------------------------------------------------------------
    // COLUMN-PARALLEL tree body (node and root)
    // ------------------------------------------------------------------
    const uint32_t start_chunk = get_arg_val<uint32_t>(3);  // always 0 for slice 0; kept for arg-layout parity
    const uint32_t num_merges = get_arg_val<uint32_t>(4);

#ifdef TOPK_TREE_ROOT
    constexpr bool is_tree_root = true;
#else
    constexpr bool is_tree_root = false;
#endif
    constexpr uint32_t slot1 = sequence_tiles;

    // An empty slice (valid_length cut) with no partners contributes nothing;
    // the writer ships the -inf scratch sequence. The root's slice 0 always
    // contains at least one element (valid_length >= 1), so it never takes
    // the empty/adopt paths and they fold away.
    const bool own_empty = !is_tree_root && (num_chunks == 0);
    if (own_empty && num_merges == 0) {
        return;
    }

#ifdef TOPK_TREE_ROOT
    compute_kernel_hw_startup(input_cb, indices_out_cb);
#else
    compute_kernel_hw_startup(input_cb, ship_values_cb);
#endif

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer recv_obj(recv_cb);
#ifdef TOPK_TREE_ROOT
    CircularBuffer indices_out_obj(indices_out_cb);
#else
    CircularBuffer ship_values_obj(ship_values_cb);
    CircularBuffer ship_indices_obj(ship_indices_cb);
#endif

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();

        if (!own_empty) {
            // Leaf reduction of this core's chunk slice. The chunk base is
            // latched at 0 and stepped with the all-constant advance
            // primitive: a runtime chunk_base cannot reach the
            // TTI-immediate config write inside the init.
            topk_xl_separate_indices_row_major_init_static<0, 0>();
            for (uint32_t c = 0; c < start_chunk; ++c) {
                topk_xl_separate_indices_row_major_advance_chunk_base<K>();
            }

            // A node whose leaf survivor ships directly (no tree merges)
            // rebuilds ASCENDING on its last DST action; the root (and any
            // survivor with merges still ahead) rebuilds DESCENDING.
            const bool ship_after_leaf = !is_tree_root && (num_merges == 0);

            const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
            process_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false);

            if (num_chunks == 1) {
                topk_xl_init<K, false>();
                topk_xl_stamp_seq_ranks<K>(slot0);
                topk_xl_rebuild<K, false>(slot0, ship_after_leaf);
            }

            for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
                const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
                process_chunk<K>(input_cb_obj, slot1, active_elements, true);

                topk_xl_init<K, false>();
                topk_xl_merge<K, false>(slot0);
                const bool last_chunk = (chunk + 1 == num_chunks);
                topk_xl_rebuild<K, false>(slot0, last_chunk && ship_after_leaf);
            }
        }

        // Tree merges: one partner sequence per winning level, in level order.
        for (uint32_t m = 0; m < num_merges; ++m) {
            const bool ship_next = !is_tree_root && (m + 1 == num_merges);
            const bool adopt = own_empty && (m == 0);
            const uint32_t dst_base = adopt ? slot0 : slot1;

            recv_obj.wait_front(sequence_tiles);
            // EXPLICIT format reconfig (bf16 -> UInt32): the preceding chunk phase
            // (topk_xl_copy_tile_init) hw-configures the unpacker for the BF16
            // input CB, and the plain _short init's state_configure is a no-op
            // in production JIT builds (the compute-kernel sentinel is compiled
            // out), so the _with_dt variant's llk reconfig is required here.
            // The reverse transition is explicit inside topk_xl_copy_tile_init.
            copy_tile_to_dst_init_short_with_dt(input_cb, recv_cb);
            for (uint32_t t = 0; t < sequence_tiles; ++t) {
                copy_tile(recv_cb, t, dst_base + t);
            }
            recv_obj.pop_front(sequence_tiles);

            topk_xl_init<K, false>();
            if (!adopt) {
                topk_xl_merge<K, false>(slot0);
            }
            topk_xl_rebuild<K, false>(slot0, ship_next /* ascending for the winner's halver */);
        }

#ifdef TOPK_TREE_ROOT
        // Root epilogue: materialize the indices exactly like the row-parallel
        // kernel (mark_neginf + rank-order transpose + pack_untilize).
        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_out_cb);
#endif

        tile_regs_commit();
        tile_regs_wait();

#ifdef TOPK_TREE_ROOT
        indices_out_obj.reserve_back(1);
        pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_out_cb);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_out_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_out_obj.push_back(1);
#else
        // Ship the survivor raw: bit-exact FP32 value tiles then UINT32 index
        // tiles; the winner's copy_tile round-trips the DST image unchanged.
        ship_values_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0, ship_values_cb, tiles_per_sequence);
        ship_values_obj.push_back(tiles_per_sequence);

        ship_indices_obj.reserve_back(tiles_per_sequence);
        pack_block(slot0 + tiles_per_sequence, ship_indices_cb, tiles_per_sequence);
        ship_indices_obj.push_back(tiles_per_sequence);
#endif

        tile_regs_release();
    }

#else  // !TOPK_TREE

    // ------------------------------------------------------------------
    // ROW-PARALLEL body (FUSED_SEGMENTED / FUSED_E2E / classic)
    // ------------------------------------------------------------------
#ifdef FUSED_SEGMENTED
    // Cross-segment survivor (unfused [v,i]) lives at tiles 0..2*tps-1; the
    // in-flight fused segment at 2*tps..3*tps-1; its incoming chunk after it.
    constexpr uint32_t slotA = sequence_tiles;
#endif
#ifdef FUSED_E2E
    // Fused survivors are half the width of unfused ones (no separate index
    // region until the final split), so the incoming chunk sits directly
    // after the survivor — the fused merge expects its operand there.
    constexpr uint32_t slot1 = tiles_per_sequence;
#else
    constexpr uint32_t slot1 = sequence_tiles;
#endif

    namespace skip = topk_large_indices_chunk_skip;

    compute_kernel_hw_startup(input_cb, indices_cb);
    pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence>(indices_cb);
#if !defined(FUSED_SEGMENTED) && !defined(FUSED_E2E)
    if constexpr (kChunkSkipEnable) {
        skip::chunk_skip_configure();
    }
#endif

    CircularBuffer input_cb_obj(input_cb);
    CircularBuffer indices_cb_obj(indices_cb);

    for (uint32_t row = 0; row < num_rows; ++row) {
        tile_regs_acquire();
#ifdef CHUNK_SKIP_TELEMETRY
        skip::telemetry_row_begin(num_chunks);
#endif

#ifdef FUSED_SEGMENTED
        constexpr uint32_t seg_cap = 32;  // 5-bit local chunk id
        const uint32_t num_segments = (num_chunks + seg_cap - 1) / seg_cap;
        for (uint32_t seg = 0; seg < num_segments; ++seg) {
            const uint32_t seg_first = seg * seg_cap;
            const uint32_t seg_end = (seg_first + seg_cap < num_chunks) ? (seg_first + seg_cap) : num_chunks;
            const uint32_t seg_last = seg_end - 1;
            const uint32_t base = (seg == 0) ? slot0 : slotA;
            const uint32_t chunkslot = base + tiles_per_sequence;
            // Segments after the first rebuild their final survivor ASCENDING:
            // the unfused cross-segment merge consumes a rank-mirrored operand
            // against the descending survivor (the tree kernels' shipped
            // rebuild-ascending pattern).
            const bool mirror_last = (seg != 0);

            // No index-tracking clear is needed after the unfused
            // cross-segment fold: every fused chunk's add_lsb_indices_init /
            // topk_xl_init<K, true> resets the whole SFPU LaneConfig register
            // (_init_sfpu_config_reg writes 0), so the unfused init's
            // index-tracking bit never survives into fused work.

            const uint32_t first_elems = (seg_first + 1 == num_chunks) ? tail_elements : K;
            copy_chunk<K>(input_cb_obj, base, first_elems);
            finish_chunk_fused_local<K>(base, false, 0);
            if (seg_first == seg_last) {
                topk_xl_rebuild<K, true>(base, mirror_last);
            }
            for (uint32_t chunk = seg_first + 1; chunk <= seg_last; ++chunk) {
                const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
                copy_chunk<K>(input_cb_obj, chunkslot, active_elements);
                finish_chunk_fused_local<K>(chunkslot, true, chunk - seg_first);
                topk_xl_merge<K, true>(base);
                topk_xl_rebuild<K, true>(base, mirror_last && (chunk == seg_last));
            }

            // One split per segment: global index = seg*32*K + local. The
            // in-place split writes indices at base + tps -- exactly the
            // unfused merge's dense operand-1 slot when base == slotA, and
            // exactly the epilogue's expected layout when base == slot0.
            topk_xl_separate_indices_row_major_global_init();
            topk_xl_separate_indices_row_major_global_base<K>(base, seg * (seg_cap * K));

            if (seg != 0) {
                topk_xl_init<K, false>();
                topk_xl_merge<K, false>(slot0);
                topk_xl_rebuild<K, false>(slot0, false);
            }
        }
#else
#ifndef FUSED_E2E
        topk_xl_separate_indices_row_major_init_static<0, 0>();
#endif

        const uint32_t first_chunk_elements = (num_chunks == 1) ? tail_elements : K;
        process_chunk<K>(input_cb_obj, slot0, first_chunk_elements, false, 0);

        if (num_chunks == 1) {
#ifdef FUSED_E2E
            topk_xl_rebuild<K, true>(slot0, false);
#else
            topk_xl_init<K, false>();
            topk_xl_stamp_seq_ranks<K>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
#endif
        }

        for (uint32_t chunk = 1; chunk < num_chunks; ++chunk) {
            const uint32_t active_elements = (chunk + 1 == num_chunks) ? tail_elements : K;
            copy_chunk<K>(input_cb_obj, slot1, active_elements);

#ifndef FUSED_E2E
            // Chunk skip is a classic-path feature: its threshold read and
            // chunk-max scratch assume the unfused DST layout, and every
            // fused-eligible k=2048 row (<= 32 chunks) sits below the skip
            // gate's first_tested_chunk anyway.
            if constexpr (kChunkSkipEnable) {
#ifdef CHUNK_SKIP_TELEMETRY
                // Telemetry variant: identical gate predicate and identical
                // per-chunk mailbox traffic on every TRISC; the decision is
                // recorded (MATH only) before the skip acts.
                if (chunk >= skip::first_tested_chunk<USER_K>()) {
                    const bool skipped = skip::chunk_skip_decide<K, USER_K>(slot1);
                    skip::telemetry_record(chunk, skipped);
                    if (skipped) {
                        topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                        continue;
                    }
                }
#else
                // Gated start: chunk >= max(2, USER_K/4). The floor of 2 is a
                // layout requirement (threshold address valid only after the
                // first merge+rebuild); USER_K/4 amortizes the test to where
                // skips are actually probable (see the header).
                if (chunk >= skip::first_tested_chunk<USER_K>() && skip::chunk_skip_decide<K, USER_K>(slot1)) {
                    // Chunk proven irrelevant (all elements strictly below
                    // the running USER_K-th survivor). The chunk was popped;
                    // only the MATH-side chunk-base bookkeeping must advance.
                    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
                    continue;
                }
#endif
            }
#endif

            finish_chunk<K>(slot1, true, chunk);

#ifdef FUSED_E2E
            topk_xl_merge<K, true>(slot0);
            topk_xl_rebuild<K, true>(slot0, false);
#else
            topk_xl_init<K, false>();
            topk_xl_merge<K, false>(slot0);
            topk_xl_rebuild<K, false>(slot0, false);
#endif
        }
#ifdef CHUNK_SKIP_TELEMETRY
        skip::telemetry_row_end<USER_K>(row, num_chunks);
#endif

#if defined(CHUNK_SKIP_DIAG) && defined(TRISC_MATH)
        // Calibration dump: raw post-rebuild values-region words of slot0.
        if (row == 0) {
            skip::chunk_skip_configure();
            ckernel::tensix_sync();
            volatile uint32_t* mm = reinterpret_cast<volatile uint32_t*>(RISCV_DEST_START_ADDR);
            for (uint32_t w = 0; w < tiles_per_sequence * 1024; ++w) {
                DPRINT("DW {} {}\n", w, mm[w]);
            }
        }
#endif

#ifdef FUSED_E2E
        // One global split per row: recover chunk_id * K + within-chunk from
        // the stamped fused words, leaving the exact unfused layout the
        // epilogue below has always consumed.
        topk_xl_separate_indices_row_major_global_init();
        topk_xl_separate_indices_row_major_global<K>(slot0);
#endif
#endif  // FUSED_SEGMENTED
        mark_neginf_indices<K>(slot0);
        materialize_index_rank_order<K>(slot0, indices_cb);

        tile_regs_commit();
        tile_regs_wait();

        indices_cb_obj.reserve_back(1);
        pack_untilize_dest<tiles_per_sequence, tiles_per_sequence>(indices_cb, 1, 0, slot0 + tiles_per_sequence);
        indices_cb_obj.push_back(1);

        tile_regs_release();
    }

#endif  // TOPK_TREE
}
