// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <
    uint32_t in_ntiles_c,
    uint32_t window_size_hw,
    bool split_reader,
    uint32_t max_out_sticks_per_core,
    uint32_t in_c,
    uint32_t in_nblocks_c,
    uint32_t max_sticks_for_reduction,
    bool is_avg_pool,
    uint32_t in_cb_id_0,
    uint32_t in_cb_id_1,  // For split-reader kernels.
    uint32_t in_scalar_cb_id_0,
    uint32_t in_scalar_cb_id_1,
    uint32_t out_cb_id,
    bool one_scalar_per_core,
    uint32_t pre_tilize_cb_id,
    bool is_output_tiled,  // true = TILED, false = ROW_MAJOR.
    bool is_output_block_format,
    bool force_max_tiles_per_reduction_4,
    uint32_t fast_tilize_cb_id>
FORCE_INLINE void pool_2d_compute_impl(uint32_t runtime_num_out_sticks) {
    constexpr uint32_t face_height = 16;
    constexpr uint32_t face_width = 16;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t tile_width = 32;
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    // fast_tilize_cb_id is a consumer-view alias of pre_tilize_cb_id (same L1 region,
    // full-tile face_geometry = {face_r_dim=16, num_faces=4}). Used as the input operand
    // to fast_tilize so the unpacker/math read the correct face count from CB metadata.

    constexpr bool last_tile_is_partial = in_c % tile_width != 0;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < tile_height || window_size_hw <= face_height) ? 2 : 4;
    // "Single partial tile per core that fits in one face": when there is only one output tile
    // per core (in_c < tile_width) and it fits in a single face (in_c <= face_width), pack just
    // one face for that tile. The host correspondingly aligns output_shard_width to face_width,
    // so the per-shard layout has no internal padding and downstream consumers (e.g.
    // sharded_to_interleaved) read contiguous data.
    constexpr bool single_partial_fits_in_face = last_tile_is_partial && in_c <= face_width;
    constexpr uint32_t num_faces_in_output_tile = single_partial_fits_in_face ? 1 : 2;
    // When the last tile has exactly face_width valid channels (channels % 32 == 16) OR the only
    // tile is partial-fits-in-one-face, pack 1 face for the last tile.
    constexpr uint32_t num_faces_in_last_output_tile =
        last_tile_is_partial && (in_c % tile_width == face_width || single_partial_fits_in_face) ? 1 : 2;

    constexpr auto reduce_op = is_avg_pool ? ckernel::PoolType::AVG : ckernel::PoolType::MAX;
    constexpr auto reduce_dim = ckernel::ReduceDim::REDUCE_COL;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time. Callers (e.g. grid_sample under fp32_dest_acc_en) can
    // also force the 4-tile limit via force_max_tiles_per_reduction_4 so each chunk fits in half-sync DEST
    // (= 4 fp32 tiles)
    // without forcing dst_full_sync_en.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION =
        (force_max_tiles_per_reduction_4 || (is_avg_pool && is_large_kernel)) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    constexpr bool neginf_srca_maxpool = reduce_op == ckernel::PoolType::MAX;
    constexpr bool zero_srca_avgpool = reduce_op == ckernel::PoolType::AVG;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     window_size_hw <= face_height && !last_tile_is_partial;

    constexpr uint32_t tilize_untilize_cb = is_output_tiled ? pre_tilize_cb_id : out_cb_id;

    DataflowBuffer in_scalar_dfb_0(in_scalar_cb_id_0);
    DataflowBuffer in_scalar_dfb_1(in_scalar_cb_id_1);
    DataflowBuffer in_dfb_0(in_cb_id_0);
    DataflowBuffer in_dfb_1(in_cb_id_1);
    DataflowBuffer out_dfb(out_cb_id);
    DataflowBuffer pre_tilize_dfb(pre_tilize_cb_id);
    DataflowBuffer fast_tilize_dfb(fast_tilize_cb_id);

    compute_kernel_hw_startup(in_cb_id_0, in_scalar_cb_id_0, tilize_untilize_cb);
    tilizeA_B_reduce_init<reduce_op, reduce_dim, neginf_srca_maxpool, zero_srca_avgpool>(
        in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter);

    pack_untilize_dest_init<max_tiles_per_iter>(tilize_untilize_cb);

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        in_scalar_dfb_0.wait_front(1);
    }

    // if max out sticks is non-zero then this will be used as the number of out sticks for every core
    // otherwise the runtime args provide the core-specific count. Pool2D uses the runtime count;
    // bilinear consumers (Rotate and GridSample) set max_out_sticks_per_core.
    uint32_t num_out_sticks_this_core = max_out_sticks_per_core ? max_out_sticks_per_core : runtime_num_out_sticks;
    uint32_t last_tile_height =
        num_out_sticks_this_core % tile_height == 0 ? tile_height : num_out_sticks_this_core % tile_height;

    uint32_t tilize_stick_counter = 0;
    uint32_t tilize_stick_total = 0;
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const bool reader0 = !(split_reader && (n & 0x1));
        const bool use_reader1_scalar = !reader0 && !one_scalar_per_core;
        const uint32_t curr_scalar_cb_id = use_reader1_scalar ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        DataflowBuffer curr_scalar_dfb = use_reader1_scalar ? in_scalar_dfb_1 : in_scalar_dfb_0;
        DataflowBuffer curr_in_dfb = reader0 ? in_dfb_0 : in_dfb_1;
        if constexpr (!one_scalar_per_core) {
            curr_scalar_dfb.wait_front(1);
        }
        if (is_output_tiled && !tilize_stick_counter) {
            out_dfb.reserve_back(in_ntiles_c);
        }
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            const uint32_t number_of_tiles = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            const uint32_t output_faces =
                (last_tile_is_partial && last_c_block &&
                 (in_c % tile_width == face_width || single_partial_fits_in_face))
                    ? (number_of_tiles - 1) * num_faces_in_output_tile + num_faces_in_last_output_tile
                    : number_of_tiles * num_faces_in_output_tile;
            if constexpr (!is_output_tiled) {
                out_dfb.reserve_back(output_faces);
            }
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce)));
                }
            }
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                curr_in_dfb.wait_front(1);
                unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    curr_in_cb_id,
                    curr_scalar_cb_id,
                    tiles_to_reduce,
                    0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/);
                for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                    reduce_tile_math<reduce_op, reduce_dim>(math_tile_idx, num_faces_in_input_tile);
                }
                curr_in_dfb.pop_front(1);
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (is_output_tiled) {
                // TILED output: accumulate sticks and perform tilization when needed
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(pre_tilize_cb_id, 1, 0);
                    pre_tilize_dfb.push_back(partial_iter_output_tiles);
                    tilize_stick_counter++;
                    tilize_stick_total++;
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(pre_tilize_cb_id, 1, 0);
                    pre_tilize_dfb.push_back(max_tiles_per_iter);
                }
                tile_regs_release();

                bool last_tile = num_out_sticks_this_core - tilize_stick_total < last_tile_height;
                if (tilize_stick_counter == tile_height || (last_tile && tilize_stick_counter == last_tile_height)) {
                    if (last_tile && last_tile_height != tile_height) {
                        pre_tilize_dfb.wait_front(last_tile_height * in_ntiles_c);
                        // if the last tile is not whole we won't have pushed enough sticks, so we need to
                        // push filler sticks to reach tile_height so the DFB pointers stay correct
                        // before calling tilize
                        uint32_t filler_stick_tiles =
                            (tile_height - last_tile_height) *
                            ((in_nblocks_c - 1) * max_tiles_per_iter + partial_iter_output_tiles);
                        pre_tilize_dfb.push_back(filler_stick_tiles);
                    }
                    PACK((pack_untilize_uninit(pre_tilize_cb_id)));

                    unpack_tilizeA_B_uninit(curr_in_cb_id);
                    pack_reconfig_data_format(out_cb_id);

                    // Hand the freshly-written L1 region off to the consumer view of the
                    // multi-format CB. pre_tilize_cb_id was pushed in tile_height*in_ntiles_c
                    // stick-pages (page_size = tile_width*nbytes); fast_tilize_cb_id sees the
                    // same bytes as in_ntiles_c full tiles (page_size = tile_size). Both views
                    // advance by the same number of bytes per round so their rd/wr pointers
                    // stay aligned. The producer-view wait_front/pop_front/reserve_back below
                    // continues to drive the producer pointer ledger.
                    fast_tilize_dfb.push_back(in_ntiles_c);
                    fast_tilize_dfb.wait_front(in_ntiles_c);

                    fast_tilize_init(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    fast_tilize_block(fast_tilize_cb_id, in_ntiles_c, out_cb_id);
                    fast_tilize_uninit(fast_tilize_cb_id, out_cb_id, in_ntiles_c);

                    out_dfb.push_back(in_ntiles_c);
                    fast_tilize_dfb.pop_front(in_ntiles_c);
                    fast_tilize_dfb.reserve_back(in_ntiles_c);
                    pre_tilize_dfb.pop_front(tile_height * in_ntiles_c);
                    pre_tilize_dfb.reserve_back(tile_height * in_ntiles_c);

                    tilize_stick_counter = 0;

                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce)));
                    // init math for reduction again since FPU gets reprogrammed by tilize
                    MATH((llk_math_reduce_init<reduce_op, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY>(
                        in_cb_id_0, in_scalar_cb_id_0)));
#ifdef ARCH_BLACKHOLE
                    // need this on BH to set swizzle bit before pack untilize dest
                    MATH((llk_math_reconfig_remap(true)));
#endif

                    if constexpr (is_output_block_format) {
                        pack_reconfig_data_format(pre_tilize_cb_id);
                    }
                    PACK((llk_pack_untilize_init<max_tiles_per_iter, max_tiles_per_iter, false, false, TILE_C_DIM>(
                        pre_tilize_cb_id)));
                }
            } else {
                // ROW_MAJOR output: pack directly to output CB
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(out_cb_id, 1, 0);
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0);
                }
                out_dfb.push_back(output_faces);
                tile_regs_release();
            }
        }
        if constexpr (!one_scalar_per_core) {
            curr_scalar_dfb.pop_front(1);
        }
    }
}
