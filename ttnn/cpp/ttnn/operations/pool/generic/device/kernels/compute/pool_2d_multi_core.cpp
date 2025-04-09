// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"

#define DEBUG_PRINT 1

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

template <uint32_t num_output_tiles, bool is_partial_tile, uint32_t split_reader, uint32_t unpA_face_r_dim>
inline void reduce_h_fused(
    const uint32_t in_cb_id_0,
    const uint32_t in_cb_id_1,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_stick_index,
    const uint32_t out_cb_id) {
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    cb_reserve_back(out_cb_id, num_output_tiles);
    const uint32_t curr_in_cb_id = (split_reader && (in_stick_index & 0x1)) ? in_cb_id_1 : in_cb_id_0;
    cb_wait_front(curr_in_cb_id, 1);

    tile_regs_acquire();
    unpack_tilizeA_B_block(
        curr_in_cb_id,
        in_scalar_cb_id,
        num_output_tiles,
        0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
        num_faces_in_tile /* unpack 1 or 2 faces ) */,
        unpA_face_r_dim);
    for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
        reduce_tile_math(c_i, num_faces_in_tile /* reduce 1 or 2 faces */);
    }
    cb_pop_front(curr_in_cb_id, 1);
    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<num_output_tiles>(
        out_cb_id, 1 /*out_subblock_h*/, 0, num_out_rows, num_faces_in_tile); /* pack 1 row (1x16 or 1x32) */
    tile_regs_release();
    cb_push_back(out_cb_id, num_output_tiles);
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(2);
    constexpr uint32_t out_h = get_compile_time_arg_val(3);
    constexpr uint32_t out_w = get_compile_time_arg_val(4);

    constexpr uint32_t split_reader = get_compile_time_arg_val(5);

    constexpr uint32_t nsticks_per_core = get_compile_time_arg_val(6);
    constexpr uint32_t in_c = get_compile_time_arg_val(7);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(8);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(10);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(11);
    constexpr uint32_t in_scalar_cb_id = get_compile_time_arg_val(12);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(13);
    constexpr uint32_t kernel_h = get_compile_time_arg_val(16);
    constexpr uint32_t kernel_w = get_compile_time_arg_val(17);
    constexpr uint32_t in_h = get_compile_time_arg_val(18);
    constexpr uint32_t in_w = get_compile_time_arg_val(19);
    constexpr uint32_t pad_h = get_compile_time_arg_val(20);
    constexpr uint32_t pad_w = get_compile_time_arg_val(21);
    constexpr uint32_t stride_h = get_compile_time_arg_val(22);
    constexpr uint32_t stride_w = get_compile_time_arg_val(23);
    constexpr uint32_t ceil_w = get_compile_time_arg_val(24);

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    uint32_t num_od_ele = get_arg_val<uint32_t>(0);
    uint32_t scalar_cnt = get_arg_val<uint32_t>(1);
    uint32_t diff_index = 0;
    uint32_t time_for_change = get_arg_val<uint32_t>(2 + diff_index);
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        in_cb_id_0, in_scalar_cb_id, max_tiles_per_iter, out_cb_id, num_faces_in_tile, window_size_hw);
    pack_untilize_dst_init_short<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_tile);
    DPRINT << "scalar cnt " << scalar_cnt << ENDL();
    for (uint32_t i = 0; i < num_od_ele; i++) {
        DPRINT << "i " << i << ENDL();
        if (i == time_for_change) {
            cb_wait_front(in_scalar_cb_id, 1);
            DPRINT << "change " << ENDL();
            if (diff_index < scalar_cnt - 1) {
                diff_index++;
                time_for_change = get_arg_val<uint32_t>(2 + diff_index);
                DPRINT << "next change coming on " << time_for_change << ENDL();
            }
        }
        // DPRINT << "i " << i << ENDL();
        //  perform the reduction over the first N - 1 whole chunks
        for (uint32_t b_i = 0; b_i < in_nblocks_c - 1; ++b_i) {
            reduce_h_fused<max_tiles_per_iter, is_partial_tile, split_reader, window_size_hw>(
                in_cb_id_0, in_cb_id_1, in_scalar_cb_id, i, out_cb_id);
        }
        // perform the reduction over the either whole or partial chunk N
        reduce_h_fused<partial_iter_output_tiles, is_partial_tile, split_reader, window_size_hw>(
            in_cb_id_0, in_cb_id_1, in_scalar_cb_id, i, out_cb_id);

        if (i + 1 == time_for_change || i == num_od_ele - 1) {
            DPRINT << "popped the old num " << ENDL();
            cb_pop_front(in_scalar_cb_id, 1);
        }
    }
}

}  // namespace NAMESPACE
