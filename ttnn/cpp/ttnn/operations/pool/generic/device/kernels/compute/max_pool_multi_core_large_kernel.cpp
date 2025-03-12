// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

template <
    uint32_t num_output_tiles,
    bool is_partial_tile,
    uint32_t max_rows_for_reduction,
    uint32_t split_reader,
    uint32_t unpA_face_r_dim>
inline void reduce_h_fused_interm(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_stick_index,
    const uint32_t interm_index,
    const uint32_t interm_cb_id) {
    constexpr uint32_t num_faces_in_input_tile = is_partial_tile ? 1 : max_rows_for_reduction < 32 ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    const uint32_t curr_in_cb_id = split_reader ? (in_cb_id + (in_stick_index & 0x1)) : in_cb_id;
    cb_wait_front(curr_in_cb_id, 1);
    tile_regs_acquire();
    unpack_tilizeA_B_block(
        curr_in_cb_id,
        in_scalar_cb_id,
        num_output_tiles,
        0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
        num_faces_in_input_tile /* unpack 1 or 2 faces ) */,
        unpA_face_r_dim);
    for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
        reduce_tile_math(c_i, num_faces_in_input_tile /* reduce 1 or 2 faces */);
    }
    cb_pop_front(curr_in_cb_id, 1);
    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<num_output_tiles>(
        interm_cb_id,
        1 /*out_subblock_h*/,
        interm_index,
        num_out_rows,
        num_faces_in_output_tile); /* pack 1 row (1x16 or 1x32) */
    tile_regs_release();
}

template <uint32_t num_output_tiles, bool is_partial_tile, uint32_t max_rows_for_reduction>
inline void reduce_h_fused(const uint32_t interm_cb_id, const uint32_t in_scalar_cb_id, const uint32_t out_cb_id) {
    constexpr uint32_t num_faces_in_input_tile = is_partial_tile ? 1 : max_rows_for_reduction < 32 ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    cb_reserve_back(out_cb_id, num_output_tiles);
    cb_wait_front(interm_cb_id, 1);
    tile_regs_acquire();
    unpack_tilizeA_B_block(
        interm_cb_id,
        in_scalar_cb_id,
        num_output_tiles,
        0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
        num_faces_in_input_tile /* unpack 1 or 2 faces ) */,
        max_rows_for_reduction);
    for (uint32_t c_i = 0; c_i < num_output_tiles; ++c_i) {
        reduce_tile_math(c_i, num_faces_in_input_tile /* reduce 1 or 2 faces */);
    }
    cb_pop_front(interm_cb_id, 1);
    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<num_output_tiles>(
        out_cb_id, 1 /*out_subblock_h*/, 0, num_out_rows, num_faces_in_output_tile); /* pack 1 row (1x16 or 1x32) */
    tile_regs_release();
    cb_push_back(out_cb_id, num_output_tiles);
}

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet.
    constexpr uint32_t in_ntiles_hw = get_compile_time_arg_val(
        0);  // note ntiles_hw will always be 1 in this kernel, when ntiles_hw > 1 the large kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(3);
    constexpr uint32_t out_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_w = get_compile_time_arg_val(5);

    constexpr uint32_t split_reader = get_compile_time_arg_val(12);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(13);
    constexpr uint32_t in_c = get_compile_time_arg_val(14);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(15);
    constexpr uint32_t max_rows_for_reduction = get_compile_time_arg_val(16);

    constexpr uint32_t in_cb_id = tt::CBIndex::c_0;  // and tt::CBIndex::c_1 for split reader
    constexpr uint32_t in_scalar_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t interm_cb_id = tt::CBIndex::c_25;

    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_input_tile = is_partial_tile ? 1 : max_rows_for_reduction < 32 ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;
    tilizeA_B_reduce_init<true>(
        in_cb_id, in_scalar_cb_id, max_tiles_per_iter, interm_cb_id, num_faces_in_input_tile, max_rows_for_reduction);

    uint32_t interm_reduction_chunks = window_size_hw / max_rows_for_reduction;
    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t i = 0; i < nsticks_per_core_by_nblocks; ++i) {
        for (uint32_t b_i = 0; b_i < in_nblocks_c - 1; b_i++) {
            // perform the intermediate reductions over the first N - 1 whole chunks
            pack_untilize_uninit(interm_cb_id);
            pack_untilize_dst_init_short<max_tiles_per_iter>(interm_cb_id, num_out_rows, num_faces_in_output_tile);
            cb_reserve_back(interm_cb_id, 1);
            for (uint32_t h = 0; h <= interm_reduction_chunks; h++) {
                reduce_h_fused_interm<
                    max_tiles_per_iter,
                    is_partial_tile,
                    max_rows_for_reduction,
                    split_reader,
                    max_rows_for_reduction>(in_cb_id, in_scalar_cb_id, i, h, interm_cb_id);
            }
            cb_push_back(interm_cb_id, 1);

            // perform the final reduction over the first N - 1 whole chunks
            pack_untilize_uninit(out_cb_id);
            pack_untilize_dst_init_short<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_output_tile);
            reduce_h_fused<max_tiles_per_iter, is_partial_tile, max_rows_for_reduction>(
                interm_cb_id, in_scalar_cb_id, out_cb_id);
        }

        // perform the intermediate reduction over chunk N (across the whole chunk even if the last chunk is partial)
        pack_untilize_uninit(interm_cb_id);
        pack_untilize_dst_init_short<max_tiles_per_iter>(interm_cb_id, num_out_rows, num_faces_in_output_tile);
        cb_reserve_back(interm_cb_id, 1);
        for (uint32_t h = 0; h <= interm_reduction_chunks; h++) {
            reduce_h_fused_interm<
                max_tiles_per_iter,
                is_partial_tile,
                max_rows_for_reduction,
                split_reader,
                max_rows_for_reduction>(in_cb_id, in_scalar_cb_id, i, h, interm_cb_id);
        }
        cb_push_back(interm_cb_id, 1);

        // perform the reduction over the either whole or partial chunk N
        pack_untilize_uninit(out_cb_id);
        pack_untilize_dst_init_short<partial_iter_output_tiles>(out_cb_id, num_out_rows, num_faces_in_output_tile);
        reduce_h_fused<partial_iter_output_tiles, is_partial_tile, max_rows_for_reduction>(
            interm_cb_id, in_scalar_cb_id, out_cb_id);
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
