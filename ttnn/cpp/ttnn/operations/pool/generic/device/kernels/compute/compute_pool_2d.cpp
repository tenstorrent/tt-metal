// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/add_int_sfpu.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#include "api/debug/dprint_tensix.h"
#include "tools/profiler/kernel_profiler.hpp"
#endif

#define ALWI inline __attribute__((always_inline))

#define FACE_HEIGHT 16
#define FACE_WIDTH 16
#define TILE_HEIGHT 32
#define TILE_WIDTH 32

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);

    constexpr uint32_t max_out_sticks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t in_c = get_compile_time_arg_val(4);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(5);
    constexpr uint32_t max_sticks_for_reduction = get_compile_time_arg_val(6);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(8);  // for split reader
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(10);
    constexpr uint32_t in_idx_cb_id = get_compile_time_arg_val(11);
    constexpr uint32_t pack_tmp_cb_id = get_compile_time_arg_val(12);
    constexpr uint32_t pack_idx_tmp_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t right_inc_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t down_left_wrap_inc_cb_id = get_compile_time_arg_val(15);
    constexpr uint32_t up_left_wrap_inc_cb_id = get_compile_time_arg_val(16);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(17);
    constexpr uint32_t out_idx_cb_id = get_compile_time_arg_val(18);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(19);
    constexpr uint32_t pre_tilize_cb_id = get_compile_time_arg_val(20);
    constexpr bool is_output_tiled = get_compile_time_arg_val(21);  // 1 = TILED, 0 = ROW_MAJOR
    constexpr bool is_output_block_format = (bool)get_compile_time_arg_val(22);
    constexpr bool return_indices = (bool)get_compile_time_arg_val(23);
    constexpr uint32_t stride_h = get_compile_time_arg_val(24);
    constexpr uint32_t stride_w = get_compile_time_arg_val(25);
    constexpr uint32_t in_h_padded = get_compile_time_arg_val(26);
    constexpr uint32_t in_w_padded = get_compile_time_arg_val(27);
    constexpr uint32_t eff_kernel_h = get_compile_time_arg_val(28);
    constexpr uint32_t eff_kernel_w = get_compile_time_arg_val(29);
    constexpr uint32_t pad_l = get_compile_time_arg_val(30);

    constexpr bool use_split_reader = split_reader && !return_indices;

    constexpr uint32_t mpwi_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    constexpr uint32_t index_dst_idx = 2;
    constexpr uint32_t inc_dst_idx = 4;
    constexpr uint32_t index_scratch_out_dst_idx = 6;

    constexpr uint32_t face_r_dim = window_size_hw < FACE_HEIGHT && !return_indices ? window_size_hw : FACE_HEIGHT;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile =
        (max_sticks_for_reduction < TILE_HEIGHT || window_size_hw <= FACE_HEIGHT) && !return_indices ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = 2;
    constexpr uint32_t num_faces_in_last_output_tile = last_tile_is_partial && in_c % TILE_WIDTH <= FACE_WIDTH ? 1 : 2;
    constexpr uint32_t num_out_sticks = 1;

    constexpr bool is_avg_pool = REDUCE_OP == PoolType::SUM;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = return_indices ? 1 : (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig = in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 &&
                                     window_size_hw <= FACE_HEIGHT && !last_tile_is_partial;

    if constexpr (!return_indices) {
        constexpr uint32_t tilize_untilize_cb = is_output_tiled ? pre_tilize_cb_id : out_cb_id;
        tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
            in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, tilize_untilize_cb, num_faces_in_input_tile, face_r_dim);
        pack_untilize_dest_init<max_tiles_per_iter>(tilize_untilize_cb, num_out_sticks, num_faces_in_output_tile);
    } else {
        unary_op_init_common(in_cb_id_0, in_cb_id_0);
        copy_tile_to_dst_init_short(in_cb_id_0);
        max_reduce_with_indices_init<ckernel::DataLayout::ROW_MAJOR>();
    }

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }
    uint32_t current_idx_col;
    uint32_t current_idx_row;
    if constexpr (return_indices) {
        const uint16_t start_row = (uint16_t)get_arg_val<uint32_t>(2);
        const uint16_t start_col = (uint16_t)get_arg_val<uint32_t>(3);
        current_idx_col = start_col;
        current_idx_row = start_row;

        cb_wait_front(right_inc_cb_id, 1);
        cb_wait_front(down_left_wrap_inc_cb_id, 1);
        cb_wait_front(up_left_wrap_inc_cb_id, 1);
        // in_idx_cb_id is populated by the reader, but this happens after the inc CBs are populated
        // so idx_tmp is protected here, and we intend to have PACK act as the sole producer, hence
        // why the reader cannot push_back here
        cb_push_back(in_idx_cb_id, 1);
    }

    // if max out sticks is non-zero then this will be used as the number of out sticks for every core
    // otherwise the runtime args are referenced for core-specific number of out sticks, for Pool2D
    // runtime args are used while for grid sample the max out sticks is set
    uint32_t num_out_sticks_this_core = max_out_sticks_per_core ? max_out_sticks_per_core : get_arg_val<uint32_t>(0);
    uint32_t last_tile_height =
        num_out_sticks_this_core % TILE_HEIGHT == 0 ? TILE_HEIGHT : num_out_sticks_this_core % TILE_HEIGHT;

    uint32_t tilize_stick_counter = 0;
    uint32_t tilize_stick_total = 0;
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const bool reader0 = !(use_split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
        }
        if (is_output_tiled && !tilize_stick_counter) {
            cb_reserve_back(out_cb_id, in_ntiles_c);
        }
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;
            const uint32_t tiles_to_reduce =
                tilize_reconfig ? (last_c_block ? partial_iter_output_tiles : max_tiles_per_iter) : max_tiles_per_iter;
            const uint32_t number_of_tiles = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            const uint32_t output_faces =
                (last_tile_is_partial && last_c_block)
                    ? (number_of_tiles - 1) * num_faces_in_output_tile + num_faces_in_last_output_tile
                    : number_of_tiles * num_faces_in_output_tile;
            if constexpr (!is_output_tiled && !return_indices) {
                cb_reserve_back(out_cb_id, output_faces);
            }
            if constexpr (tilize_reconfig) {
                if (first_c_block || last_c_block) {
                    UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce, num_faces_in_input_tile, face_r_dim, 1)));
                }
            }
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                cb_wait_front(curr_in_cb_id, 1);
                if constexpr (return_indices) {
                    reconfig_data_format_srca(curr_in_cb_id);
                    copy_tile(curr_in_cb_id, mpwi_cb_tile_idx, data_dst_idx);

                    if (first_c_block) {
                        cb_wait_front(in_idx_cb_id, 1);
                    }
                    reconfig_data_format_srca(in_idx_cb_id);
                    copy_tile(in_idx_cb_id, mpwi_cb_tile_idx, index_dst_idx);
                    if (last_c_block) {
                        cb_pop_front(in_idx_cb_id, 1);

                        // update the current index column
                        if (current_idx_col + stride_w + eff_kernel_w > in_w_padded) {
                            // we reached the right edge, wrap down and to the left
                            current_idx_col = 0;
                            if (current_idx_row + stride_h + eff_kernel_h > in_h_padded) {
                                // we reached the bottom right corner, wrap to the top and to the left
                                current_idx_row = 0;
                                copy_tile(up_left_wrap_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                            } else {
                                current_idx_row += stride_h;
                                copy_tile(down_left_wrap_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                            }
                        } else {
                            // we are still in the same row, move to the right
                            current_idx_col += stride_w;
                            copy_tile(right_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        }

                        // we allow overflow here for negative values as this only occurs in padding regions
                        add_int_tile_init();
                        add_uint16_tile(index_dst_idx, inc_dst_idx, index_scratch_out_dst_idx);

                        max_reduce_with_indices_init<ckernel::DataLayout::ROW_MAJOR>();
                    }

                    // the max_reduce_with_indices LLK function only supports kernel_size=9, pending
                    // https://github.com/tenstorrent/tt-metal/issues/28141 but, since for return_indices the in_cb is
                    // oversized (equal to 1 tile), and since this CB is filled with padding values in the beginning of
                    // the data movement kernel, it is possible to still use max_reduce_with_indices with kernel sizes
                    // smaller than 9 as the excess sticks are just filled with padding values
                    constexpr uint32_t max_mpwi_kernel_size = window_size_hw <= 9 ? 9 : 32;
                    max_reduce_with_indices<max_mpwi_kernel_size, ckernel::DataLayout::ROW_MAJOR>(
                        data_dst_idx, index_dst_idx);
                } else {
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
                }
                cb_pop_front(curr_in_cb_id, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (!return_indices) {
                if constexpr (is_output_tiled) {
                    // TILED output: accumulate sticks and perform tilization when needed
                    if (last_c_block) {
                        pack_untilize_dest<partial_iter_output_tiles>(
                            pre_tilize_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                        cb_push_back(pre_tilize_cb_id, partial_iter_output_tiles);
                        tilize_stick_counter++;
                        tilize_stick_total++;
                    } else {
                        pack_untilize_dest<max_tiles_per_iter>(
                            pre_tilize_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                        cb_push_back(pre_tilize_cb_id, max_tiles_per_iter);
                    }
                    tile_regs_release();

                    bool last_tile = num_out_sticks_this_core - tilize_stick_total < last_tile_height;
                    if (tilize_stick_counter == TILE_HEIGHT ||
                        (last_tile && tilize_stick_counter == last_tile_height)) {
                        if (last_tile && last_tile_height != TILE_HEIGHT) {
                            cb_wait_front(pre_tilize_cb_id, last_tile_height * in_ntiles_c);
                            // if the last tile is not whole we won't have pushed enough sticks, so we need to
                            // push some filler sticks to reach TILE_HEIGHT to make sure the CB pointers are correct
                            // before calling tilize
                            uint32_t filler_stick_tiles =
                                (TILE_HEIGHT - last_tile_height) *
                                ((in_nblocks_c - 1) * max_tiles_per_iter + partial_iter_output_tiles);
                            cb_push_back(pre_tilize_cb_id, filler_stick_tiles);
                        }
                        cb_wait_front(pre_tilize_cb_id, TILE_HEIGHT * in_ntiles_c);
                        PACK((pack_untilize_uninit(pre_tilize_cb_id)));

                        unpack_tilizeA_B_uninit(curr_in_cb_id);
                        pack_reconfig_data_format(out_cb_id);

                        fast_tilize_init(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
                        fast_tilize_block(pre_tilize_cb_id, in_ntiles_c, out_cb_id);
                        fast_tilize_uninit(pre_tilize_cb_id, out_cb_id);

                        cb_push_back(out_cb_id, in_ntiles_c);
                        cb_pop_front(pre_tilize_cb_id, TILE_HEIGHT * in_ntiles_c);
                        cb_reserve_back(pre_tilize_cb_id, TILE_HEIGHT * in_ntiles_c);

                        if constexpr (is_output_block_format) {
                            pack_reconfig_data_format(pre_tilize_cb_id);
                        }

                        tilize_stick_counter = 0;

                        UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                            in_cb_id_0, in_scalar_cb_id_0, tiles_to_reduce, num_faces_in_input_tile, face_r_dim, 1)));
                        // init math for reduction again since FPU gets reprogrammed by tilize
                        MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>()));
#ifdef ARCH_BLACKHOLE
                        // need this on BH to set swizzle bit before pack untilize dest
                        MATH((llk_math_reconfig_remap(true)));
#endif
                        PACK((llk_pack_untilize_init<max_tiles_per_iter, max_tiles_per_iter, false, false, TILE_C_DIM>(
                            pre_tilize_cb_id, 1, num_faces_in_output_tile)));
                    }
                } else {
                    // ROW_MAJOR output: pack directly to output CB
                    if (last_c_block) {
                        pack_untilize_dest<partial_iter_output_tiles>(
                            out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                    } else {
                        pack_untilize_dest<max_tiles_per_iter>(
                            out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                    }
                    cb_push_back(out_cb_id, output_faces);
                    tile_regs_release();
                }
            } else {
                cb_reserve_back(pack_tmp_cb_id, 1);
                pack_reconfig_data_format(pack_tmp_cb_id);
                pack_tile<true>(data_dst_idx, pack_tmp_cb_id, mpwi_cb_tile_idx);
                cb_push_back(pack_tmp_cb_id, 1);

                cb_reserve_back(pack_idx_tmp_cb_id, 1);
                pack_reconfig_data_format(pack_idx_tmp_cb_id);
                pack_tile<true>(index_dst_idx, pack_idx_tmp_cb_id, mpwi_cb_tile_idx);
                cb_push_back(pack_idx_tmp_cb_id, 1);

                if (last_c_block) {
                    cb_reserve_back(in_idx_cb_id, 1);
                    pack_tile<true>(index_scratch_out_dst_idx, in_idx_cb_id, mpwi_cb_tile_idx);
                    cb_push_back(in_idx_cb_id, 1);
                }
                tile_regs_release();
            }
        }
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
}

}  // namespace NAMESPACE
