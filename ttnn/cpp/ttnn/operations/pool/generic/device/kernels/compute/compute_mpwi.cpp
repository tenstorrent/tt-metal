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
    constexpr uint32_t intra_kernel_right_inc_cb_id = get_compile_time_arg_val(31);
    constexpr uint32_t intra_kernel_down_left_wrap_inc_cb_id = get_compile_time_arg_val(32);
    constexpr uint32_t kernel_h = get_compile_time_arg_val(33);
    constexpr uint32_t kernel_w = get_compile_time_arg_val(34);

    constexpr uint32_t mpwi_cb_tile_idx = 0;
    constexpr uint32_t data_dst_idx = 0;
    constexpr uint32_t index_dst_idx = 2;
    constexpr uint32_t inc_dst_idx = 4;
    constexpr uint32_t index_scratch_out_dst_idx = 6;
    constexpr uint32_t index_temp_dst_idx = 7;  // only used for large kernels

    constexpr uint32_t face_r_dim = FACE_HEIGHT;
    constexpr bool last_tile_is_partial = in_c % TILE_WIDTH != 0;
    constexpr uint32_t num_faces_in_input_tile = 4;
    constexpr uint32_t num_faces_in_output_tile = 2;
    constexpr uint32_t num_faces_in_last_output_tile = last_tile_is_partial && in_c % TILE_WIDTH <= FACE_WIDTH ? 1 : 2;
    constexpr uint32_t num_out_sticks = 1;

    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr bool is_large_kernel = window_size_hw > max_sticks_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 1;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX, "Only supports REDUCE_OP = MAX");
    constexpr bool neginf_srca_maxpool = true;
    constexpr bool zero_srca_avgpool = false;

    constexpr uint32_t remaining_elems = window_size_hw % max_sticks_for_reduction;
    constexpr w_chunks = kernel_w % max_rows_for_reduction == 0 ? kernel_w / max_rows_for_reduction
                                                                : kernel_w / max_rows_for_reduction + 1;
    constexpr uint32_t interm_reduction_chunks = return_indices    ? w_chunks * kenrel_h
                                                 : remaining_elems ? window_size_hw / max_sticks_for_reduction + 1
                                                                   : window_size_hw / max_sticks_for_reduction;

    cb_wait_front(in_scalar_cb_id_0, 1);

    uint32_t current_idx_col;
    uint32_t current_idx_row;
    const uint16_t start_row = (uint16_t)get_arg_val<uint32_t>(1);
    const uint16_t start_col = (uint16_t)get_arg_val<uint32_t>(2);
    current_idx_col = start_col;
    current_idx_row = start_row;

    uint32_t sticks_per_chunk = 0;
    cb_wait_front(right_inc_cb_id, 1);
    cb_wait_front(down_left_wrap_inc_cb_id, 1);
    cb_wait_front(up_left_wrap_inc_cb_id, 1);
    cb_wait_front(in_idx_cb_id, 1);
    reconfig_data_format_srca(in_idx_cb_id);
    copy_tile(in_idx_cb_id, mpwi_cb_tile_idx, index_dst_idx);  // move the initial indexes to DST where they will live
    if (is_large_kernel) {
        sticks_per_chunk = kernel_w <= max_sticks_for_reduction ? kernel_w : max_sticks_for_reduction;
        cb_wait_front(intra_kernel_right_inc_cb_id, 1);
        cb_wait_front(intra_kernel_down_left_wrap_inc_cb_id, 1);
        copy_dest_values_init();
        copy_dest_values(index_dst_idx, index_temp_dst_idx);  // make a copy of the initial indexes for large kernel use
    }

    unary_op_init_common(in_cb_id_0, in_cb_id_0);
    copy_tile_to_dst_init_short(in_cb_id_0);
    max_reduce_with_indices_init<ckernel::DataLayout::ROW_MAJOR>();

    // if max out sticks is non-zero then this will be used as the number of out sticks for every core
    // otherwise the runtime args are referenced for core-specific number of out sticks, for Pool2D
    // runtime args are used while for grid sample the max out sticks is set
    uint32_t num_out_sticks_this_core = max_out_sticks_per_core ? max_out_sticks_per_core : get_arg_val<uint32_t>(0);

    uint32_t tilize_stick_counter = 0;
    uint32_t tilize_stick_total = 0;
    for (uint32_t n = 0; n < num_out_sticks_this_core; ++n) {
        const uint32_t curr_scalar_cb_id = in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = in_cb_id_0;
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const bool last_c_block = c_i == in_nblocks_c - 1;
            const bool first_c_block = c_i == 0;

            tile_regs_acquire();
            uint32_t intra_kernel_h = 0;
            uint32_t intra_kernel_w = 0;
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                bool first_chunk = chunk == 0;
                bool last_chunk = chunk == interm_reduction_chunks - 1;
                cb_wait_front(curr_in_cb_id, 1);

                reconfig_data_format_srca(curr_in_cb_id);
                copy_tile(curr_in_cb_id, mpwi_cb_tile_idx, data_dst_idx);

                // increments happen between every chunk within a C block, and between C blocks
                bool increment_needed = false;
                if (last_c_block && last_chunk) {  // increment for the next kernel position
                    increment_needed = true;
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
                } else if (is_large_kernel) {  // only need to increment within C block if multiple chunks
                    if (last_chunk) {          // reset to the initial indexes for this C block
                        copy_dest_values_init();
                        copy_dest_values(index_temp_dst_idx, index_dst_idx);
                    } else {  // increment for the next chunk within the same C block
                        increment_needed = true;
                        if (intra_kernel_w + sticks_per_chunk < kernel_w) {  // move right in this row
                            intra_kernel_w += sticks_per_chunk;
                            copy_tile(intra_kernel_right_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        } else {  // move down to the next row
                            intra_kernel_w = 0;
                            intra_kernel_h += 1;
                            copy_tile(intra_kernel_down_left_wrap_inc_cb_id, mpwi_cb_tile_idx, inc_dst_idx);
                        }
                    }
                }

                if (increment_needed) {
                    // we allow overflow here for negative values as this only occurs in padding regions
                    add_int_tile_init();
                    add_uint16_tile(index_dst_idx, inc_dst_idx, index_scratch_out_dst_idx);
                    copy_dest_values_init();
                    copy_dest_values(index_scratch_out_dst_idx, index_dst_idx);
                    if (is_large_kernel) {
                        copy_dest_values(index_scratch_out_dst_idx, index_temp_dst_idx);
                    }

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

                cb_pop_front(curr_in_cb_id, 1);
            }
            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(pack_tmp_cb_id, 1);
            pack_reconfig_data_format(pack_tmp_cb_id);
            pack_tile<true>(data_dst_idx, pack_tmp_cb_id, mpwi_cb_tile_idx);
            cb_push_back(pack_tmp_cb_id, 1);

            cb_reserve_back(pack_idx_tmp_cb_id, 1);
            pack_reconfig_data_format(pack_idx_tmp_cb_id);
            pack_tile<true>(index_dst_idx, pack_idx_tmp_cb_id, mpwi_cb_tile_idx);
            cb_push_back(pack_idx_tmp_cb_id, 1);

            tile_regs_release();
        }
    }
}

}  // namespace NAMESPACE
