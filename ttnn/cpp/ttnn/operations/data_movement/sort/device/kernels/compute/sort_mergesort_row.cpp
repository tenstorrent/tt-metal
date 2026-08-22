// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include <cstdint>

/*
Mergesort row engine (issue #33492 roadmap): a full sort of one row per core
built from the TopK XL SFPU kernels, replacing the tile-pair bitonic outer
schedule whose cost is per TILE-ROW rather than per element.

W <= 4096 (num_chunks <= 2): everything happens inside one 32-bit full-sync
DEST window — the shipped fast path:
  1. Leaf(s): stream the row in K=2048-element chunks; each chunk is copied
     into DEST, stamped with LINEAR position tags fused into the value words'
     free low 16 bits ([bf16 | u16], sign-conditioned complement so ties break
     exactly as torch.sort(stable=True)), and locally sorted by the fused
     bitonic network. Tag order == element order, so stability is structural —
     no comparator, no index tiles ride the network.
  2. Merge level (W = 4096): the two opposite-direction leaves form a bitonic
     2K-sequence; one both-halves merge + a rebuild per half leaves the fully
     sorted 4096-run across DEST tiles 0..3.
  3. Split: each K-run separates into a stripped bf16 value region (in place)
     and a UINT32 index region (the tag IS the global row index).
  4. Materialize: transpose_dest + pack_untilize emit rank-ordered pages
     (16-element slices in face-pair order; the writer permutes slices on
     the way out, exactly like topk_large_indices' writer).

W >= 8192 (num_chunks in {4..32}): the L1-STAGED MERGE TREE. Sorted fused
runs park between merge levels in two ping-pong Float32-format run buffers
(run_a / run_b, compute-produced AND compute-consumed; raw 32-bit transport —
the only format that preserves the lo16 tags, a bf16 pack would RNE-round
them away). Passes, per row:
  - LEAF pass: for each chunk pair, the shipped W=4096 body (copy + stamp +
    two opposite-direction local sorts + both-halves merge + rebuild x2) with
    a raw pack_tile tail into run_a — the first merge level folds into the
    leaf for free. Runs alternate direction per the bitonic mergesort
    recursion: dir(run r) = ascending ^ (popcount(r) & 1).
  - CROSS passes (per merge level, chunk strides s = m .. 2 where m = input
    run length in chunks): load chunk c -> DEST tiles 0-1 and chunk c+s ->
    tiles 2-3; the both-halves merge's stride-2048 window distance IS the
    stride-s element distance because the chunks sit s apart; raw-pack both
    chunks back to the SAME chunk slots in the other run buffer.
  - FINAL pass (per level): per 4096-block, merge_both_halves (the stride-2048
    stage) + rebuild per half; intermediate levels raw-pack back, the LAST
    level fuses with EMIT: separate indices into DEST tiles 4-7 +
    transpose_dest + pack_untilize per block — the shipped W=4096 tail, one
    block at a time, so the 8-tile DEST budget is honored at every W.
Credit discipline: whole-pass wait_front / reserve_back on the run buffers
(random access within the pass via rd-relative copy_tile tile indices and
pack_tile<true> absolute offsets), push_back + pop_front at pass boundaries —
those credits are what order TRISC2's L1 write-back against TRISC0's
read-ahead on the next pass (tile_regs orders DEST ownership, not L1).

Stability across levels: tags are stamped ONCE at the leaf with the GLOBAL
row position (chunk_base + lane position, chunk_base < 65536 — the u16
identity ceiling, W <= 65536). Every merge consumes two disjoint
origin-contiguous runs, so tag order within a value-tie class stays origin
order at every level; no re-stamp exists or is needed.

DEST layout (tile indices), K = 2048, fast path:
  W=2048: [values 0-1 | indices 2-3]
  W=4096: [run0 values 0-1 | run1 values 2-3 | run0 indices 4-5 | run1 indices 6-7]
Staged path: tiles 0-3 = the active 4096-word fused window; tiles 4-7 only
used by the per-block emit (split-out indices), exactly the W=4096 layout.
*/

namespace {
constexpr uint32_t K = 2048;
constexpr uint32_t TILES_PER_RUN = 2;
}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);

    constexpr uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr bool descending = get_arg(args::descending);
    static_assert(
        num_chunks == 1 || num_chunks == 2 || num_chunks == 4 || num_chunks == 8 || num_chunks == 16 ||
            num_chunks == 32,
        "mergesort row engine: num_chunks must be a power of two in [1, 32] (W = 2048 .. 65536)");

    // Global sort direction: leaf A sorts in the global direction; leaf B in
    // the opposite one so their concatenation is bitonic for the merge level.
    constexpr bool ascending = !descending;
    // The stamp complements the tag bits of the tie class the sign-magnitude
    // compare would otherwise reverse: negative words for ascending sorts,
    // non-negative words for descending ones.
    constexpr bool complement_non_negative = descending;

    DataflowBuffer input_stage_dfb(dfb::input_stage);
    DataflowBuffer values_out_dfb(dfb::values_out);
    DataflowBuffer indices_out_dfb(dfb::indices_out);

    compute_kernel_hw_startup(dfb::input_stage, dfb::values_out);

#ifndef MERGESORT_STAGED
    {
        static_assert(num_chunks <= 2, "fast-path build requires num_chunks <= 2");
        // -------- Fast path: the whole row fits one DEST window --------
        // Index regions sit one full W of value tiles after the runs.
        constexpr uint32_t indices_base_tile = num_chunks * TILES_PER_RUN;
        constexpr uint32_t indices_dst_offset = indices_base_tile * 64;

        for (uint32_t row = 0; row < num_rows; ++row) {
            tile_regs_acquire();

            // ---- Leaf A: chunk 0 -> DEST tiles [0, TILES_PER_RUN) ----
            input_stage_dfb.wait_front(TILES_PER_RUN);
            topk_xl_copy_tile_init(dfb::input_stage);
            topk_xl_copy_tile<K>(dfb::input_stage, 0, 0, K);
            input_stage_dfb.pop_front(TILES_PER_RUN);

            topk_xl_add_lsb_indices_init();
            topk_xl_add_linear_indices<K>(0, 0, complement_non_negative);
            topk_xl_init<K, true>();
            topk_xl_local_sort<K>(0, ascending);

            if constexpr (num_chunks == 2) {
                // ---- Leaf B: chunk 1 -> DEST tiles [TILES_PER_RUN, 2*TILES_PER_RUN) ----
                input_stage_dfb.wait_front(TILES_PER_RUN);
                topk_xl_copy_tile_init(dfb::input_stage);
                topk_xl_copy_tile<K>(dfb::input_stage, TILES_PER_RUN, 0, K);
                input_stage_dfb.pop_front(TILES_PER_RUN);

                topk_xl_add_lsb_indices_init();
                topk_xl_add_linear_indices<K>(TILES_PER_RUN, K, complement_non_negative);
                topk_xl_init<K, true>();
                topk_xl_local_sort<K>(TILES_PER_RUN, !ascending);

                // ---- Merge level: both halves kept, then a rebuild per half ----
                topk_xl_init<K, true>();
                topk_xl_merge_both_halves<K>(0, ascending);
                topk_xl_rebuild<K, true>(0, ascending);
                topk_xl_rebuild<K, true>(TILES_PER_RUN, ascending);
            }

            // ---- Split each run into stripped values (in place) + u32 indices ----
            topk_xl_separate_indices_linear_init();
            topk_xl_separate_indices_linear<K, indices_dst_offset>(0, complement_non_negative);
            if constexpr (num_chunks == 2) {
                topk_xl_separate_indices_linear<K, indices_dst_offset>(TILES_PER_RUN, complement_non_negative);
            }

            // ---- Rank-order materialization (same pipeline as topk_large_indices) ----
            transpose_dest_init<true, false>(dfb::indices_out);
            for (uint32_t t = 0; t < 2 * num_chunks * TILES_PER_RUN; ++t) {
                transpose_dest<true, false>(t);
            }

            tile_regs_commit();
            tile_regs_wait();

            // Values: one K-element page per run (fp32 DEST words -> bf16 pages).
            pack_untilize_dest_init<TILES_PER_RUN, TILES_PER_RUN>(dfb::values_out);
            for (uint32_t run = 0; run < num_chunks; ++run) {
                values_out_dfb.reserve_back(1);
                pack_untilize_dest<TILES_PER_RUN, TILES_PER_RUN>(dfb::values_out, 1, 0, run * TILES_PER_RUN);
                values_out_dfb.push_back(1);
            }
            pack_untilize_uninit(dfb::values_out);

            // Indices: one K-element page per run (raw u32 words through a
            // Float32-format page — a 16-bit pack would round the index bits).
            pack_untilize_dest_init<TILES_PER_RUN, TILES_PER_RUN>(dfb::indices_out);
            for (uint32_t run = 0; run < num_chunks; ++run) {
                indices_out_dfb.reserve_back(1);
                pack_untilize_dest<TILES_PER_RUN, TILES_PER_RUN>(
                    dfb::indices_out, 1, 0, indices_base_tile + run * TILES_PER_RUN);
                indices_out_dfb.push_back(1);
            }
            pack_untilize_uninit(dfb::indices_out);

            tile_regs_release();
        }
    }
#else
    {
        static_assert(num_chunks >= 4, "staged build requires num_chunks >= 4");
        // -------- Staged path: L1 merge tree over ping-pong run buffers --------
        DataflowBuffer run_a_dfb(dfb::run_a);
        DataflowBuffer run_b_dfb(dfb::run_b);
        constexpr uint32_t total_tiles = num_chunks * TILES_PER_RUN;
        // Per-block emit splits indices into tiles 4-7 (4 value tiles below).
        constexpr uint32_t emit_indices_dst_offset = 4 * 64;

        // dir(run r) = ascending ^ (popcount(r) & 1): the classic bitonic
        // mergesort direction recursion (left child keeps the parent's
        // direction, right child inverts), closed-form at every level.
        const auto run_dir = [](uint32_t r) -> bool {
            return ascending ^ static_cast<bool>(__builtin_popcount(r) & 1);
        };

        for (uint32_t row = 0; row < num_rows; ++row) {
            // ---- LEAF pass (folds the first merge level): input -> run_a ----
            run_a_dfb.reserve_back(total_tiles);
            for (uint32_t p = 0; p < num_chunks / 2; ++p) {
                const bool d = run_dir(p);
                tile_regs_acquire();

                input_stage_dfb.wait_front(TILES_PER_RUN);
                topk_xl_copy_tile_init(dfb::input_stage);
                topk_xl_copy_tile<K>(dfb::input_stage, 0, 0, K);
                input_stage_dfb.pop_front(TILES_PER_RUN);
                topk_xl_add_lsb_indices_init();
                topk_xl_add_linear_indices<K>(0, (2 * p) * K, complement_non_negative);
                topk_xl_init<K, true>();
                topk_xl_local_sort<K>(0, d);

                input_stage_dfb.wait_front(TILES_PER_RUN);
                topk_xl_copy_tile_init(dfb::input_stage);
                topk_xl_copy_tile<K>(dfb::input_stage, TILES_PER_RUN, 0, K);
                input_stage_dfb.pop_front(TILES_PER_RUN);
                topk_xl_add_lsb_indices_init();
                topk_xl_add_linear_indices<K>(TILES_PER_RUN, (2 * p + 1) * K, complement_non_negative);
                topk_xl_init<K, true>();
                topk_xl_local_sort<K>(TILES_PER_RUN, !d);

                topk_xl_init<K, true>();
                topk_xl_merge_both_halves<K>(0, d);
                topk_xl_rebuild<K, true>(0, d);
                topk_xl_rebuild<K, true>(TILES_PER_RUN, d);

                tile_regs_commit();
                tile_regs_wait();
                // Raw pack: fused [bf16|u16] words must ride Float32 pages
                // bit-preserved (FP32-mode narrowing or a bf16 pack would
                // destroy the tags).
                pack_reconfig_data_format(dfb::run_a);
                for (uint32_t t = 0; t < 4; ++t) {
                    pack_tile<true>(t, dfb::run_a, 4 * p + t);
                }
                tile_regs_release();
            }
            run_a_dfb.push_back(total_tiles);

            // ---- MERGE LEVELS ----
            bool cur_is_a = true;  // which buffer holds the current level's runs
            uint32_t run_chunks = 2;
            while (run_chunks < num_chunks) {
                const uint32_t out_chunks = run_chunks * 2;
                const bool is_last_level = (out_chunks == num_chunks);

                // -- CROSS passes: chunk strides s = run_chunks .. 2 --
                for (uint32_t s = run_chunks; s >= 2; s >>= 1) {
                    const uint32_t src_cb = cur_is_a ? dfb::run_a : dfb::run_b;
                    const uint32_t dst_cb = cur_is_a ? dfb::run_b : dfb::run_a;
                    DataflowBuffer& src_dfb = cur_is_a ? run_a_dfb : run_b_dfb;
                    DataflowBuffer& dst_dfb = cur_is_a ? run_b_dfb : run_a_dfb;
                    src_dfb.wait_front(total_tiles);
                    dst_dfb.reserve_back(total_tiles);
                    // Pass-level init hygiene: within a cross pass the ops are
                    // homogeneous (datacopy loads + one both-halves merge + raw
                    // packs), so unpack/pack reconfig and the topk_xl ADDR_MOD/MOP
                    // programming hoist to the pass boundary.
                    reconfig_data_format_srca(src_cb);
                    copy_tile_to_dst_init_short(src_cb);
                    pack_reconfig_data_format(dst_cb);
                    topk_xl_init<K, true>();
                    for (uint32_t r = 0; r < num_chunks / out_chunks; ++r) {
                        const bool d = run_dir(r);
                        const uint32_t base = r * out_chunks;
                        for (uint32_t blk = 0; blk < out_chunks; blk += 2 * s) {
                            for (uint32_t i = 0; i < s; ++i) {
                                const uint32_t c1 = base + blk + i;
                                const uint32_t c2 = c1 + s;
                                tile_regs_acquire();
                                copy_tile(src_cb, 2 * c1, 0);
                                copy_tile(src_cb, 2 * c1 + 1, 1);
                                copy_tile(src_cb, 2 * c2, 2);
                                copy_tile(src_cb, 2 * c2 + 1, 3);
                                topk_xl_merge_both_halves<K>(0, d);
                                tile_regs_commit();
                                tile_regs_wait();
                                pack_tile<true>(0, dst_cb, 2 * c1);
                                pack_tile<true>(1, dst_cb, 2 * c1 + 1);
                                pack_tile<true>(2, dst_cb, 2 * c2);
                                pack_tile<true>(3, dst_cb, 2 * c2 + 1);
                                tile_regs_release();
                            }
                        }
                    }
                    dst_dfb.push_back(total_tiles);
                    src_dfb.pop_front(total_tiles);
                    cur_is_a = !cur_is_a;
                }

                // -- FINAL pass: stride-2048 merge + rebuild per 4096-block;
                //    the LAST level fuses with the emit tail --
                {
                    const uint32_t src_cb = cur_is_a ? dfb::run_a : dfb::run_b;
                    const uint32_t dst_cb = cur_is_a ? dfb::run_b : dfb::run_a;
                    DataflowBuffer& src_dfb = cur_is_a ? run_a_dfb : run_b_dfb;
                    DataflowBuffer& dst_dfb = cur_is_a ? run_b_dfb : run_a_dfb;
                    src_dfb.wait_front(total_tiles);
                    if (!is_last_level) {
                        dst_dfb.reserve_back(total_tiles);
                        pack_reconfig_data_format(dst_cb);
                    }
                    reconfig_data_format_srca(src_cb);
                    copy_tile_to_dst_init_short(src_cb);
                    topk_xl_init<K, true>();
                    for (uint32_t r = 0; r < num_chunks / out_chunks; ++r) {
                        const bool d = run_dir(r);
                        for (uint32_t c = r * out_chunks; c < (r + 1) * out_chunks; c += 2) {
                            tile_regs_acquire();
                            if (is_last_level) {
                                // The emit tail reconfigures unpack/pack per block;
                                // re-establish the datacopy state for the loads.
                                reconfig_data_format_srca(src_cb);
                                copy_tile_to_dst_init_short(src_cb);
                            }
                            copy_tile(src_cb, 2 * c, 0);
                            copy_tile(src_cb, 2 * c + 1, 1);
                            copy_tile(src_cb, 2 * c + 2, 2);
                            copy_tile(src_cb, 2 * c + 3, 3);
                            if (is_last_level) {
                                topk_xl_init<K, true>();
                            }
                            topk_xl_merge_both_halves<K>(0, d);
                            topk_xl_rebuild<K, true>(0, d);
                            topk_xl_rebuild<K, true>(TILES_PER_RUN, d);
                            if (!is_last_level) {
                                tile_regs_commit();
                                tile_regs_wait();
                                pack_tile<true>(0, dst_cb, 2 * c);
                                pack_tile<true>(1, dst_cb, 2 * c + 1);
                                pack_tile<true>(2, dst_cb, 2 * c + 2);
                                pack_tile<true>(3, dst_cb, 2 * c + 3);
                                tile_regs_release();
                            } else {
                                // ---- EMIT (fused with the last final pass):
                                // the shipped W=4096 tail, per 4096-block ----
                                topk_xl_separate_indices_linear_init();
                                topk_xl_separate_indices_linear<K, emit_indices_dst_offset>(0, complement_non_negative);
                                topk_xl_separate_indices_linear<K, emit_indices_dst_offset>(
                                    TILES_PER_RUN, complement_non_negative);

                                transpose_dest_init<true, false>(dfb::indices_out);
                                for (uint32_t t = 0; t < 8; ++t) {
                                    transpose_dest<true, false>(t);
                                }

                                tile_regs_commit();
                                tile_regs_wait();

                                pack_untilize_dest_init<TILES_PER_RUN, TILES_PER_RUN>(dfb::values_out);
                                for (uint32_t run = 0; run < 2; ++run) {
                                    values_out_dfb.reserve_back(1);
                                    pack_untilize_dest<TILES_PER_RUN, TILES_PER_RUN>(
                                        dfb::values_out, 1, 0, run * TILES_PER_RUN);
                                    values_out_dfb.push_back(1);
                                }
                                pack_untilize_uninit(dfb::values_out);

                                pack_untilize_dest_init<TILES_PER_RUN, TILES_PER_RUN>(dfb::indices_out);
                                for (uint32_t run = 0; run < 2; ++run) {
                                    indices_out_dfb.reserve_back(1);
                                    pack_untilize_dest<TILES_PER_RUN, TILES_PER_RUN>(
                                        dfb::indices_out, 1, 0, 4 + run * TILES_PER_RUN);
                                    indices_out_dfb.push_back(1);
                                }
                                pack_untilize_uninit(dfb::indices_out);

                                tile_regs_release();
                            }
                        }
                    }
                    if (!is_last_level) {
                        dst_dfb.push_back(total_tiles);
                        cur_is_a = !cur_is_a;
                    }
                    src_dfb.pop_front(total_tiles);
                }

                run_chunks = out_chunks;
            }
        }
    }
#endif  // MERGESORT_STAGED
}
