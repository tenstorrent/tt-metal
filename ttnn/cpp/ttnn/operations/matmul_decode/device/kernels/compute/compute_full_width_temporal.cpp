// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

using std::uint32_t;

// deep-plan_14 Lever 2 -- WIDTH-temporal k_stream compute kernel.
//
// Streams A in G_temporal K-slices (reader_full_width_temporal.cpp) instead of gathering
// the full A (which busts L1/CBCAP for SigLIP fc2 K=4320 / VLM down K=16384). K-OUTER loop
// over G_temporal slices; the running K-partial is accumulated IN-DST across slices and
// packed ONCE at the end (preserves fp32 reduction order). The per-output-tile DST is seeded
// from the running fp32 accumulator (c_4) before each slice's matmul_block calls accumulate
// on top of it -- out_h=1, out_w<=4 so the DST footprint is <=4 tiles (inside the fp32 cap).
//
// out_w-only fat-fill (out_h forced 1 -- OQ1 de-risk): B (in1) is [K_tiles x N_tiles_per_core]
// row-major so the out_w in1 tiles for a fixed K-row are contiguous. Within-slice A: the reader
// publishes each slice CONTIGUOUS at the front of the in0 CB -- tile (m, kc) at m*k_slice+kc.
//
// Loop order: OUTPUT-tile-group OUTER, K-slice INNER. Each output tile-group keeps one DST
// acquire across ALL G_temporal slices (in-DST accumulation), so c_4 carries the partial
// across the (output-group) iterations: we re-consume the streamed slices once per group, so
// the reader streams the full A G_groups times. For fc2/VLM-down npc=1 -> ONE output group
// (N_tiles_per_core small, the M loop is the group axis) ... see structure below.

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t k_slice_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(4);
    constexpr uint32_t G_temporal = get_compile_time_arg_val(5);
    constexpr uint32_t DST_CAP = get_compile_time_arg_val(6);  // 8 (fp32 dest) / 16 (bf16 dest)

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;   // streamed A slice
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;   // resident full-K B shard
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;   // output shard

    constexpr uint32_t slice_num_tiles = M_tiles * k_slice_tiles;
    constexpr uint32_t out_num_tiles = M_tiles * N_tiles_per_core;

    cb_wait_front(in1_cb_id, K_tiles * N_tiles_per_core);
    cb_reserve_back(out_cb_id, out_num_tiles);

    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, 1, 1);

    // ---- K-OUTER loop over the G_temporal streamed K-slices ----
    // The full output rectangle [M_tiles x N_tiles_per_core] is held in DST across ALL slices.
    // out_h=1 and out_w<=4 with N_tiles_per_core small (fc2 npc=1 -> N_tpc=1; VLM-down npc=1 ->
    // N_tpc=1): out_num_tiles = M_tiles tiles. fc2 M_tiles=8, VLM-down M_tiles=9 -> <=9 fp32 DST
    // tiles. (DST holds up to 8 fp32 / 16 bf16; M_tiles=9 packs in two acquire groups below.)
    //
    // We hold one DST acquire spanning all slices for a group of <= DST_CAP output tiles, then
    // pack once. Group the M-rows into DST_CAP-sized chunks (out_w fattens N within a row).
    // out tiles per M-row = N_tiles_per_core; rows per DST group so rows*N_tpc <= DST_CAP.
    constexpr uint32_t rows_per_group =
        (N_tiles_per_core >= DST_CAP) ? 1u : (DST_CAP / N_tiles_per_core);

    for (uint32_t mt0 = 0; mt0 < M_tiles; mt0 += rows_per_group) {
        const uint32_t rows = (mt0 + rows_per_group <= M_tiles) ? rows_per_group : (M_tiles - mt0);
        // Re-init the matmul engine at the start of each output group (the previous group's
        // tile_regs_release leaves the math/unpack state to be re-established for this group's
        // first matmul_block).
        mm_block_init_short(in0_cb_id, in1_cb_id, false, out_subblock_w, 1, 1);
        tile_regs_acquire();
        // Accumulate this group's output tiles over ALL K-slices (in-DST, no pack between).
        for (uint32_t s = 0; s < G_temporal; ++s) {
            cb_wait_front(in0_cb_id, slice_num_tiles);
            const uint32_t k_global0 = s * k_slice_tiles;
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t mt = mt0 + r;
                for (uint32_t nc0 = 0; nc0 < N_tiles_per_core; nc0 += out_subblock_w) {
                    const uint32_t dst0 = (r * N_tiles_per_core + nc0);
                    for (uint32_t kc = 0; kc < k_slice_tiles; ++kc) {
                        const uint32_t in0_tile = mt * k_slice_tiles + kc;
                        const uint32_t in1_tile = (k_global0 + kc) * N_tiles_per_core + nc0;
                        matmul_block(
                            in0_cb_id, in1_cb_id, in0_tile, in1_tile, dst0, false,
                            out_subblock_w, 1, 1);
                    }
                }
            }
            cb_pop_front(in0_cb_id, slice_num_tiles);
            // The reader re-streams the full A for each output group (cb refills); only the
            // LAST group leaves the stream drained. (For fc2/VLM-down rows_per_group covers
            // ALL M-rows in ONE group, so each slice is consumed exactly once -- see factory.)
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t nc = 0; nc < N_tiles_per_core; ++nc) {
                const uint32_t dst = r * N_tiles_per_core + nc;
                const uint32_t out_slot = (mt0 + r) * N_tiles_per_core + nc;
                pack_tile<true>(dst, out_cb_id, out_slot);
            }
        }
        tile_regs_release();
    }

    cb_push_back(out_cb_id, out_num_tiles);
}
