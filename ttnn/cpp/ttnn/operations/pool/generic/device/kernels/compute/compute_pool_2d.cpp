// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/common.h"
#include "tools/profiler/kernel_profiler.hpp"

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
    constexpr uint32_t weight_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t mul_cb_id = get_compile_time_arg_val(14);

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

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;
    uint32_t num_pages_to_8 = 8 / in_ntiles_c;

    // tilize reconfiguration can be beneficial when we have a wide tensor with a non MAX_TILES_PER_REDUCTION number of
    // C tiles, but we only use it when the window size fits within a face such that the tilize can be done only on the
    // rows populated with data, otherwise we need to call clear_out_tiles between reconfigs to avoid untilizing junk
    // data which is much slower than just untilizing the entire MAX_TILES_PER_REDUCTION
    constexpr bool tilize_reconfig =
        in_nblocks_c > 1 && in_ntiles_c % MAX_TILES_PER_REDUCTION != 0 && window_size_hw <= 16;
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        mul_cb_id, in_scalar_cb_id_0, max_tiles_per_iter, out_cb_id, num_faces_in_input_tile, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_sticks, num_faces_in_output_tile);

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;

    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_sticks_for_reduction + 1 : window_size_hw / max_sticks_for_reduction;

    // wait for initialization to complete
    if constexpr (one_scalar_per_core) {
        cb_wait_front(in_scalar_cb_id_0, 1);
    }

    uint32_t iters = (nsticks_per_core_by_nblocks + num_pages_to_8 - 1) / num_pages_to_8;

    uint32_t sticks_left = nsticks_per_core_by_nblocks;
    uint32_t curr_in_cb_id = in_cb_id_0;

    while (sticks_left) {
        DeviceZoneScopedN("iteration");
        const uint32_t curr_scalar_cb_id = in_scalar_cb_id_0;
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
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                UNPACK((llk_unpack_tilize_uninit(curr_in_cb_id)));
                // UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(curr_in_cb_id, weight_cb_id)));
                PACK((pack_untilize_uninit(mul_cb_id)));
                mul_tiles_init(curr_in_cb_id, weight_cb_id);

                tile_regs_acquire();

                while (num_pages_to_8 > sticks_left) {
                    num_pages_to_8--;
                }

                DPRINT << "hajde " << num_pages_to_8 << ENDL();

                for (uint32_t j = 0; j < num_pages_to_8; j++) {
                    DeviceZoneScopedN("wait and mul tiles");
                    cb_wait_front(curr_in_cb_id, tiles_to_reduce);

                    for (uint32_t i = 0; i < tiles_to_reduce; ++i) {
                        mul_tiles(curr_in_cb_id, weight_cb_id, i, 0, j * tiles_to_reduce + i);
                    }
                    cb_pop_front(curr_in_cb_id, tiles_to_reduce);

                    sticks_left--;
                    curr_in_cb_id = (curr_in_cb_id == in_cb_id_0) ? in_cb_id_1 : in_cb_id_0;
                    tensix_sync();
                }

                tile_regs_commit();

                tile_regs_wait();
                cb_reserve_back(mul_cb_id, tiles_to_reduce * num_pages_to_8);
                for (uint32_t j = 0; j < num_pages_to_8; j++) {
                    DeviceZoneScopedN("pack tiles");
                    for (uint32_t i = 0; i < tiles_to_reduce; ++i) {
                        pack_tile(j * tiles_to_reduce + i, mul_cb_id, i);
                    }
                    tensix_sync();
                }
                cb_push_back(mul_cb_id, tiles_to_reduce * num_pages_to_8);  // TR2 zabo

                tile_regs_release();

                UNPACK((llk_unpack_tilizeA_B_init<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    mul_cb_id, curr_scalar_cb_id, tiles_to_reduce, num_faces_in_output_tile, face_r_dim, 1)));
                MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>()));
                PACK((llk_pack_untilize_init<tiles_to_reduce, tiles_to_reduce, false, false, TILE_C_DIM>(
                    out_cb_id, 1, num_faces_in_output_tile)));
                PACK((llk_init_packer_dest_offset_registers<true, false>()));

                tile_regs_acquire();

                for (uint32_t j = 0; j < num_pages_to_8; j++) {
                    DeviceZoneScopedN("unpack and reduce tiles");
                    cb_wait_front(mul_cb_id, tiles_to_reduce);
                    unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                        mul_cb_id, curr_scalar_cb_id, tiles_to_reduce, 0, num_faces_in_input_tile, face_r_dim);
                    for (uint32_t math_tile_idx = 0; math_tile_idx < tiles_to_reduce; ++math_tile_idx) {
                        reduce_tile_math(j * tiles_to_reduce + math_tile_idx, num_faces_in_input_tile);
                    }
                    cb_pop_front(mul_cb_id, tiles_to_reduce);
                    tensix_sync();
                }
                tile_regs_commit();

                tile_regs_wait();

                for (uint32_t i = 0; i < num_pages_to_8; i++) {
                    DeviceZoneScopedN("pack output tiles");
                    cb_reserve_back(out_cb_id, tiles_to_reduce);
                    pack_untilize_dest<partial_iter_output_tiles>(
                        out_cb_id,
                        1 /*out_subblock_h*/,
                        0,
                        num_out_sticks,
                        num_faces_in_output_tile,
                        i * tiles_to_reduce); /* pack 1 row (1x16 or 1x32) */
                    cb_push_back(out_cb_id, tiles_to_reduce);
                    tensix_sync();
                }

                tile_regs_release();
            }
            // if (last_c_block) {
            //     pack_untilize_dest<partial_iter_output_tiles>(
            //         out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
            //     cb_push_back(out_cb_id, partial_iter_output_tiles);
            // } else {
            //     pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, num_out_sticks, num_faces_in_output_tile);
            //     cb_push_back(out_cb_id, max_tiles_per_iter);
            // }
        }
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
        tensix_sync();
    }
}

}  // namespace NAMESPACE
