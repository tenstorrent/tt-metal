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

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

#define TILE_WIDTH 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16

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
    constexpr uint32_t in_idx_cb_id_0 = get_compile_time_arg_val(9);
    constexpr uint32_t in_idx_cb_id_1 = get_compile_time_arg_val(10);  // for split reader
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(11);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(12);
    constexpr uint32_t tile_tmp_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t tile_idx_tmp_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(15);
    constexpr uint32_t out_idx_cb_id = get_compile_time_arg_val(16);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(17);
    constexpr bool return_indices = (bool)get_compile_time_arg_val(18);

    constexpr uint32_t topk_output_tiles = 1;
    constexpr uint32_t topk_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    constexpr uint32_t index_dst_idx = 2;

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
    constexpr bool pack_untilize_reinit = last_tile_is_partial && in_ntiles_c > 1;
    if constexpr (!return_indices) {
        tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
            in_cb_id_0, in_scalar_cb_id_0, max_tiles_per_iter, out_cb_id, num_faces_in_input_tile, face_r_dim);
        pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_sticks, num_faces_in_output_tile);
    } else {
        unary_op_init_common(in_cb_id_0, in_cb_id_0);
        tilize_init_no_pack(in_cb_id_0, topk_output_tiles);
        if constexpr (!pack_untilize_reinit) {
            const uint32_t output_faces =
                last_tile_is_partial ? num_faces_in_last_output_tile : num_faces_in_output_tile;
            pack_untilize_dest_init<topk_output_tiles>(out_cb_id, num_out_sticks, output_faces);
        }

        // this can be done here because we do not use the SFPU for anything else so it does not get reprogrammed
        // if you use the sfpu for other operations, you need to call this to reprogram the sfpu
        max_reduce_with_indices_init();
    }

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }

    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        const bool reader0 = !(split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        const uint32_t curr_in_idx_cb_id = !reader0 ? in_idx_cb_id_1 : in_idx_cb_id_0;
        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);
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
            cb_reserve_back(out_cb_id, output_faces);
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
                    cb_wait_front(curr_in_idx_cb_id, 1);

                    tilize_init_short_with_dt_no_pack(curr_in_cb_id, curr_in_idx_cb_id, topk_output_tiles);
                    tilize_block_no_pack(curr_in_idx_cb_id, topk_output_tiles, index_dst_idx, topk_cb_tile_idx);
                    tilize_uninit_with_dt_no_pack(curr_in_idx_cb_id, curr_in_cb_id);
                    tilize_init_short_with_dt_no_pack(curr_in_idx_cb_id, curr_in_cb_id, topk_output_tiles);
                    tilize_block_no_pack(curr_in_cb_id, topk_output_tiles, data_dst_idx, topk_cb_tile_idx);
                    tilize_uninit_with_dt_no_pack(curr_in_cb_id, curr_in_idx_cb_id);

                    max_reduce_with_indices<window_size_hw>(data_dst_idx, index_dst_idx);

                    cb_pop_front(curr_in_idx_cb_id, 1);
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
                // for non-return index version we only have 1 output tensor therefore we can let pack_untilize_dest
                // overflow by always packing num_faces_in_output_tile even for the last tile which may have less
                if (last_c_block) {
                    pack_untilize_dest<partial_iter_output_tiles>(
                        out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                } else {
                    pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
                }
            } else {
                if constexpr (pack_untilize_reinit) {
                    tensix_sync();
                    pack_untilize_dest_init<topk_output_tiles>(out_cb_id, num_out_sticks, output_faces);
                    tensix_sync();
                }

                pack_reconfig_data_format(out_cb_id);
                pack_untilize_dest<topk_output_tiles, topk_output_tiles, false, false, TILE_C_DIM, data_dst_idx>(
                    out_cb_id, 1, 0, num_out_sticks, output_faces);
                pack_reconfig_data_format(out_idx_cb_id);
                pack_untilize_dest<topk_output_tiles, topk_output_tiles, false, false, TILE_C_DIM, index_dst_idx>(
                    out_idx_cb_id, 1, 0, num_out_sticks, output_faces);

                if constexpr (pack_untilize_reinit) {
                    tensix_sync();
                    pack_untilize_uninit(out_cb_id);
                    tensix_sync();
                }
            }
            cb_push_back(out_cb_id, output_faces);
            if constexpr (return_indices) {
                cb_push_back(out_idx_cb_id, output_faces);
            }
            tile_regs_release();
        }
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
}

}  // namespace NAMESPACE
