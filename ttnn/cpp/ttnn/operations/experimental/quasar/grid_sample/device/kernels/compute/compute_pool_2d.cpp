// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// grid_sample-local Metal 2.0 copy of ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/compute_pool_2d.cpp.
// Bound by the quasar grid_sample and quasar rotate bilinear factories; the pool/generic original keeps serving
// its legacy consumers unchanged. The kernel logic is the original's; only the argument / buffer plumbing differs.
//
// Conditionally bound DFBs are gated at the preprocessor (the generated dfb:: tokens exist only when bound):
//   SPLIT_READER  - dfb::in_1 and dfb::scalar_1 (second reader's input and scalar buffers)
//   OUTPUT_TILED  - dfb::pre_tilize and dfb::fast_tilize (tiled-output staging)
//   OUT_STICKS_PER_CORE_FROM_RTA - the per-core out-stick count arrives as a runtime arg instead of the
//                                   max_out_sticks_per_core compile-time arg

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/add_int_sfpu.h"
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "experimental/kernel_args.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint_tensix.h"
#include "api/debug/kernel_profiler.hpp"
#endif

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16  // allow-redefined-helper: kept as in the pool/generic original this file copies
#define FACE_WIDTH 16   // allow-redefined-helper: kept as in the pool/generic original this file copies
#define TILE_HEIGHT 32  // allow-redefined-helper: kept as in the pool/generic original this file copies
#define TILE_WIDTH 32   // allow-redefined-helper: kept as in the pool/generic original this file copies

void kernel_main() {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_arg(args::in_ntiles_c);
    constexpr uint32_t window_size_hw = get_arg(args::window_size_hw);

    constexpr uint32_t split_reader = get_arg(args::split_reader);

    constexpr uint32_t max_out_sticks_per_core = get_arg(args::max_out_sticks_per_core);
    constexpr uint32_t in_c = get_arg(args::in_c);
    constexpr uint32_t in_nblocks_c = get_arg(args::in_nblocks_c);
    constexpr uint32_t max_sticks_for_reduction = get_arg(args::max_sticks_for_reduction);

    // dfb::in_0 / dfb::scalar_0 are fed by reader 0; dfb::in_1 / dfb::scalar_1 (SPLIT_READER only) by reader 1.
    constexpr bool one_scalar_per_core = get_arg(args::one_scalar_per_core);
    constexpr bool is_output_tiled = get_arg(args::is_output_tiled);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_arg(args::is_output_block_format);
    // dfb::fast_tilize is a consumer-view alias of dfb::pre_tilize (same L1 region,
    // full-tile face_geometry = {face_r_dim=16, num_faces=4}). Used as the input operand
    // to fast_tilize so the unpacker/math read the correct face count from DFB metadata.

    constexpr bool use_split_reader = split_reader;

    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_HEIGHT || window_size_hw <= FACE_HEIGHT) ? 2 : 4;
    // "Single partial tile per core that fits in one face": when there is only one output tile
    // per core (in_c < TILE_WIDTH) and it fits in a single face (in_c <= FACE_WIDTH), pack just
    // one face for that tile. The host correspondingly aligns output_shard_width to FACE_WIDTH,
    // so the per-shard layout has no internal padding and downstream consumers (e.g.
    // sharded_to_interleaved) read contiguous data.
    constexpr bool single_partial_fits_in_face = last_tile_is_partial && in_c <= FACE_WIDTH;
    constexpr uint32_t num_faces_in_output_tile = single_partial_fits_in_face ? 1 : 2;
    // When the last tile has exactly FACE_WIDTH valid channels (channels % 32 == 16) OR the only
    // tile is partial-fits-in-one-face, pack 1 face for the last tile.
    constexpr uint32_t num_faces_in_last_output_tile =
        last_tile_is_partial && (in_c % TILE_WIDTH == FACE_WIDTH || single_partial_fits_in_face) ? 1 : 2;

    constexpr bool is_avg_pool = REDUCE_OP == PoolType::AVG;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time. Callers (e.g. grid_sample under fp32_dest_acc_en) can
    // also force the 4-tile limit via the force_max_tiles_per_reduction_4 arg so each chunk fits in half-sync
    // DEST (= 4 fp32 tiles) without forcing dst_full_sync_en.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr bool force_max_tiles_per_reduction_4 = get_arg(args::force_max_tiles_per_reduction_4);
    constexpr uint32_t MAX_TILES_PER_REDUCTION =
        (force_max_tiles_per_reduction_4 || (is_avg_pool && is_large_kernel)) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::AVG, "Only supports REDUCE_OP = MAX or AVG");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::AVG) ? true : false;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     window_size_hw <= FACE_HEIGHT && !last_tile_is_partial;

#ifdef OUTPUT_TILED
    constexpr uint32_t tilize_untilize_dfb_id = dfb::pre_tilize;
#else
    constexpr uint32_t tilize_untilize_dfb_id = dfb::out;
#endif

    DataflowBuffer in_scalar_dfb_0(dfb::scalar_0);
    DataflowBuffer in_dfb_0(dfb::in_0);
#ifdef SPLIT_READER
    DataflowBuffer in_scalar_dfb_1(dfb::scalar_1);
    DataflowBuffer in_dfb_1(dfb::in_1);
#endif
    DataflowBuffer out_dfb(dfb::out);
#ifdef OUTPUT_TILED
    DataflowBuffer pre_tilize_dfb(dfb::pre_tilize);
    DataflowBuffer fast_tilize_dfb(dfb::fast_tilize);
#endif

    compute_kernel_hw_startup(dfb::in_0, dfb::scalar_0, tilize_untilize_dfb_id);
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(dfb::in_0, dfb::scalar_0, max_tiles_per_iter);

    pack_untilize_dest_init<max_tiles_per_iter>(tilize_untilize_dfb_id);

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        in_scalar_dfb_0.wait_front(1);
    }

    // if max out sticks is non-zero then this will be used as the number of out sticks for every core
    // otherwise the runtime args are referenced for core-specific number of out sticks, for Pool2D
    // runtime args are used while for grid sample the max out sticks is set
#ifdef OUT_STICKS_PER_CORE_FROM_RTA
    uint32_t num_out_sticks_this_core =
        max_out_sticks_per_core ? max_out_sticks_per_core : get_arg(args::num_out_sticks_this_core);
#else
    uint32_t num_out_sticks_this_core = max_out_sticks_per_core;
#endif
    uint32_t last_tile_height =
        num_out_sticks_this_core % TILE_HEIGHT == 0 ? TILE_HEIGHT : num_out_sticks_this_core % TILE_HEIGHT;

    uint32_t tilize_stick_counter = 0;
    uint32_t tilize_stick_total = 0;
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const bool reader0 = !(use_split_reader && (n & 0x1));
        const bool use_reader1_scalar = !reader0 && !one_scalar_per_core;
#ifdef SPLIT_READER
        const uint32_t curr_scalar_dfb_id =
            use_reader1_scalar ? static_cast<uint32_t>(dfb::scalar_1) : static_cast<uint32_t>(dfb::scalar_0);
        const uint32_t curr_in_dfb_id = !reader0 ? static_cast<uint32_t>(dfb::in_1) : static_cast<uint32_t>(dfb::in_0);
        DataflowBuffer curr_scalar_dfb = use_reader1_scalar ? in_scalar_dfb_1 : in_scalar_dfb_0;
        DataflowBuffer curr_in_dfb = reader0 ? in_dfb_0 : in_dfb_1;
#else
        // Without a split reader every stick comes from reader 0.
        const uint32_t curr_scalar_dfb_id = dfb::scalar_0;
        const uint32_t curr_in_dfb_id = dfb::in_0;
        DataflowBuffer curr_scalar_dfb = in_scalar_dfb_0;
        DataflowBuffer curr_in_dfb = in_dfb_0;
#endif
        if constexpr (!one_scalar_per_core) {
            curr_scalar_dfb.wait_front(1);
        }
        if (is_output_tiled && !tilize_stick_counter) {
            out_dfb.reserve_back(in_ntiles_c);
#ifdef OUTPUT_TILED
            // The producer view is pushed a stick at a time across the whole round, so it is
            // reserved here, before the round's first push.
            pre_tilize_dfb.reserve_back(TILE_HEIGHT * in_ntiles_c);
#endif
        }
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            const uint32_t number_of_tiles = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            const uint32_t output_faces =
                (last_tile_is_partial && last_c_block &&
                 (in_c % TILE_WIDTH == FACE_WIDTH || single_partial_fits_in_face))
                    ? (number_of_tiles - 1) * num_faces_in_output_tile + num_faces_in_last_output_tile
                    : number_of_tiles * num_faces_in_output_tile;
            if constexpr (!is_output_tiled) {
                out_dfb.reserve_back(output_faces);
            }
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        dfb::in_0, dfb::scalar_0, tiles_to_reduce)));
                }
            }
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                curr_in_dfb.wait_front(1);
                unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    curr_in_dfb_id,
                    curr_scalar_dfb_id,
                    tiles_to_reduce,
                    0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/);
                for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                    reduce_tile_math<REDUCE_OP, REDUCE_DIM>(math_tile_idx, num_faces_in_input_tile);
                }
                curr_in_dfb.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
#ifdef OUTPUT_TILED
            if constexpr (is_output_tiled) {
                // TILED output: accumulate sticks and perform tilization when needed
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(dfb::pre_tilize, 1, 0);
                    pre_tilize_dfb.push_back(partial_iter_output_tiles);
                    tilize_stick_counter++;
                    tilize_stick_total++;
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(dfb::pre_tilize, 1, 0);
                    pre_tilize_dfb.push_back(max_tiles_per_iter);
                }
                tile_regs_release();

                bool last_tile = num_out_sticks_this_core - tilize_stick_total < last_tile_height;
                if (tilize_stick_counter == TILE_HEIGHT || (last_tile && tilize_stick_counter == last_tile_height)) {
                    if (last_tile && last_tile_height != TILE_HEIGHT) {
                        pre_tilize_dfb.wait_front(last_tile_height * in_ntiles_c);
                        // if the last tile is not whole we won't have pushed enough sticks, so we need to
                        // push some filler sticks to reach TILE_HEIGHT to make sure the DFB pointers are correct
                        // before calling tilize
                        uint32_t filler_stick_tiles =
                            (TILE_HEIGHT - last_tile_height) *
                            ((in_nblocks_c - 1) * max_tiles_per_iter + partial_iter_output_tiles);
                        pre_tilize_dfb.push_back(filler_stick_tiles);
                    }
                    pack_untilize_uninit(dfb::pre_tilize);

                    unpack_tilizeA_B_uninit(curr_in_dfb_id);
                    pack_reconfig_data_format(dfb::out);

                    // Hand the freshly-written L1 region off to the consumer view of the
                    // multi-format buffer. dfb::pre_tilize was pushed in TILE_HEIGHT*in_ntiles_c
                    // stick-pages (page_size = TILE_WIDTH*nbytes); dfb::fast_tilize sees the
                    // same bytes as in_ntiles_c full tiles (page_size = tile_size). Both views
                    // advance by the same number of bytes per round so their rd/wr pointers
                    // stay aligned. The producer-view wait_front/pop_front/reserve_back below
                    // continues to drive the producer pointer ledger.
                    fast_tilize_dfb.reserve_back(in_ntiles_c);
                    fast_tilize_dfb.push_back(in_ntiles_c);
                    fast_tilize_dfb.wait_front(in_ntiles_c);

                    fast_tilize_init(dfb::fast_tilize, in_ntiles_c, dfb::out);
                    fast_tilize_block(dfb::fast_tilize, in_ntiles_c, dfb::out);
                    fast_tilize_uninit(dfb::fast_tilize, dfb::out, in_ntiles_c);

                    out_dfb.push_back(in_ntiles_c);
                    fast_tilize_dfb.pop_front(in_ntiles_c);
                    pre_tilize_dfb.pop_front(TILE_HEIGHT * in_ntiles_c);

                    tilize_stick_counter = 0;

                    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
                        dfb::in_0, dfb::scalar_0, tiles_to_reduce);
                    if constexpr (is_output_block_format) {
                        pack_reconfig_data_format(dfb::pre_tilize);
                    }
                    pack_untilize_dest_init<max_tiles_per_iter, max_tiles_per_iter, false /*narrow_row*/, TILE_C_DIM>(
                        dfb::pre_tilize);
                }
            } else
#endif  // OUTPUT_TILED
            {
                // ROW_MAJOR output: pack directly to output DFB
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(dfb::out, 1, 0);
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(dfb::out, 1, 0);
                }
                out_dfb.push_back(output_faces);
                tile_regs_release();
            }
        }
        if constexpr (!one_scalar_per_core) {
            curr_scalar_dfb.pop_front(1);
        }
    }

    // Pairs with the wait_front before the loop.
    if constexpr (one_scalar_per_core) {
        in_scalar_dfb_0.pop_front(1);
    }
}
