// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include <cstdint>

/*
Mergesort row engine (issue #33492 roadmap): a full sort of one row per core
built from the TopK XL SFPU kernels, replacing the tile-pair bitonic outer
schedule whose cost is per TILE-ROW rather than per element.

Per row (all within one 32-bit full-sync DEST window):
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
     (16-element slices in face-pair order; the writer permutes slices on the
     way out, exactly like topk_large_indices' writer).

DEST layout (tile indices), K = 2048:
  W=2048: [values 0-1 | indices 2-3]
  W=4096: [run0 values 0-1 | run1 values 2-3 | run0 indices 4-5 | run1 indices 6-7]
Ascending sorts put the min half in run0, descending the max half, so run r
always materializes output elements [r*K, (r+1)*K).
*/

namespace {
constexpr uint32_t K = 2048;
constexpr uint32_t TILES_PER_RUN = 2;
}  // namespace

void kernel_main() {
    const uint32_t num_rows = get_arg(args::num_rows);

    constexpr uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr bool descending = get_arg(args::descending);
    static_assert(num_chunks == 1 || num_chunks == 2, "mergesort row engine: 1 or 2 chunks (W = 2048 or 4096)");

    // Global sort direction: leaf A sorts in the global direction; leaf B in
    // the opposite one so their concatenation is bitonic for the merge level.
    constexpr bool ascending = !descending;
    // The stamp complements the tag bits of the tie class the sign-magnitude
    // compare would otherwise reverse: negative words for ascending sorts,
    // non-negative words for descending ones.
    constexpr bool complement_non_negative = descending;
    // Index regions sit one full W of value tiles after the runs.
    constexpr uint32_t indices_base_tile = num_chunks * TILES_PER_RUN;
    constexpr uint32_t indices_dst_offset = indices_base_tile * 64;

    DataflowBuffer input_stage_dfb(dfb::input_stage);
    DataflowBuffer values_out_dfb(dfb::values_out);
    DataflowBuffer indices_out_dfb(dfb::indices_out);

    compute_kernel_hw_startup(dfb::input_stage, dfb::values_out);

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
