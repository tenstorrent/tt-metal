// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "tt-metalium/constants.hpp"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/common.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

#define TILE_HEIGHT 32
#define TILE_WIDTH 32

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);  // for split reader
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(10);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(11);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(12);
    constexpr uint32_t tmp_cb_id = get_compile_time_arg_val(13);
    constexpr bool is_output_tiled = get_compile_time_arg_val(14);  // 1 = TILED, 0 = ROW_MAJOR

    constexpr uint32_t face_r_dim = window_size_hw < 16 ? window_size_hw : 16;
    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_input_tile = is_partial_tile                                           ? 1
                                                 : (max_sticks_for_reduction < 32 || window_size_hw <= 16) ? 2
                                                                                                           : 4;
    constexpr uint32_t num_faces_in_output_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_sticks = 1;

    constexpr bool is_avg_pool = REDUCE_OP == PoolType::SUM;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    DPRINT << "max_tiles_per_iter: " << max_tiles_per_iter << ENDL();
    DPRINT << "partial_iter_output_tiles: " << partial_iter_output_tiles << ENDL();

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig =
        in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 && window_size_hw <= 16;

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }

    if constexpr (is_output_tiled) {
        // TILED output: initialize with temp CB for tilization
        tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
            in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, tmp_cb_id, num_faces_in_input_tile, face_r_dim);
        pack_untilize_dest_init<max_tiles_per_iter>(tmp_cb_id, num_out_sticks, num_faces_in_output_tile);
    } else {
        // ROW_MAJOR output: initialize directly to output CB (original behavior)
        tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
            in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, out_cb_id, num_faces_in_input_tile, face_r_dim);
        pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_sticks, num_faces_in_output_tile);
    }
    uint32_t tilize_stick_counter = 0;
    uint32_t row = 0;
    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        DPRINT << "STICK NO " << n << ENDL();
        const bool reader0 = !(split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
        }
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce, num_faces_in_input_tile, face_r_dim, 1)));
                }
            }
            if constexpr (is_output_tiled) {
                // TILED output: initialize pack for temp CB
                if (last_c_block) {
                    PACK((llk_pack_untilize_init<
                          partial_iter_output_tiles,
                          partial_iter_output_tiles,
                          false,
                          false,
                          TILE_C_DIM>(tmp_cb_id, 1, num_faces_in_output_tile)));
                } else {
                    PACK((llk_pack_untilize_init<max_tiles_per_iter, max_tiles_per_iter, false, false, TILE_C_DIM>(
                        tmp_cb_id, 1, num_faces_in_output_tile)));
                }
            }
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                cb_wait_front(curr_in_cb_id, 1);
                unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    curr_in_cb_id,
                    curr_scalar_cb_id,
                    tiles_to_reduce,
                    0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
                    num_faces_in_input_tile,
                    face_r_dim);
                for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                    reduce_tile_math(math_tile_idx, num_faces_in_input_tile);
                }
                cb_pop_front(curr_in_cb_id, 1);
            }

            // dprint_tensix_dest_reg(0);
            // dprint_tensix_dest_reg(1);
            // dprint_tensix_dest_reg(2);
            // dprint_tensix_dest_reg(3);
            // dprint_tensix_dest_reg(4);
            // dprint_tensix_dest_reg(5);
            // dprint_tensix_dest_reg(6);
            // dprint_tensix_dest_reg(7);
            tile_regs_commit();
            tile_regs_wait();

            if constexpr (is_output_tiled) {
                // TILED output: accumulate sticks in temp CB for later tilization
                if (tilize_stick_counter < TILE_HEIGHT) {
                    if (last_c_block) {
                        pack_untilize_dest<1>(tmp_cb_id, 1, row, num_out_sticks, num_faces_in_output_tile);
                        row += partial_iter_output_tiles;
                        tilize_stick_counter++;
                    } else {
                        pack_untilize_dest<1>(tmp_cb_id, 1, row, num_out_sticks, num_faces_in_output_tile);
                        row += max_tiles_per_iter;
                    }
                }
            } else {
                // ROW_MAJOR output: pack directly to output CB (original behavior)
                if (last_c_block) {
                    cb_reserve_back(out_cb_id, partial_iter_output_tiles);
                    pack_untilize_dest<partial_iter_output_tiles>(
                        out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                    cb_push_back(out_cb_id, partial_iter_output_tiles);
                } else {
                    cb_reserve_back(out_cb_id, max_tiles_per_iter);
                    pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                    cb_push_back(out_cb_id, max_tiles_per_iter);
                }
            }
            tile_regs_release();

            if constexpr (is_output_tiled) {
                // TILED output: perform tilization when TILE_HEIGHT sticks are accumulated
                if (tilize_stick_counter == TILE_HEIGHT) {
                    // Finalize temp CB and perform tilization to output CB
                    cb_push_back(tmp_cb_id, in_ntiles_c);
                    PACK((pack_untilize_uninit(tmp_cb_id)));

                    PACK(pack_reconfig_data_format(out_cb_id));
                    reconfig_data_format_srca(tmp_cb_id);

                    tilize_init(tmp_cb_id, in_ntiles_c, out_cb_id);
                    cb_wait_front(tmp_cb_id, in_ntiles_c);
                    cb_reserve_back(out_cb_id, in_ntiles_c);
                    tilize_block(tmp_cb_id, in_ntiles_c, out_cb_id);
                    cb_pop_front(tmp_cb_id, in_ntiles_c);
                    cb_push_back(out_cb_id, in_ntiles_c);
                    tilize_uninit(tmp_cb_id, out_cb_id);

                    // Reinitialize for next tile
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce, num_faces_in_input_tile, face_r_dim, 1)));
                    tilize_stick_counter = 0;
                    row = 0;
                }
            }
        }
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
        DPRINT << "END ITERATION " << n << ENDL();
    }
}

}  // namespace NAMESPACE
