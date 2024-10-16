// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"

// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);  // Leftover row size after multicore processing
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t Kt = get_compile_time_arg_val(9);
    constexpr uint32_t logk = get_compile_time_arg_val(10);
    constexpr uint32_t logWt = get_compile_time_arg_val(11);

    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;
    // init pack, compute and unpack

    init_sfpu(input_cb_index);
    ckernel::topk_tile_init();

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        cb_wait_front(input_cb_index, Wt);
        cb_wait_front(index_cb_index, Wt);

        // have to use a different buffer than input_cb_index and index_cb_index as we pop/reserve/wait/push on tiles
        // for all the in-place operations we will end up racing with reader if both kernels use cb apis on the same
        // buffer (input_cb_index/index_cb_index)
        pack_reconfig_data_format(input_transposed_cb_index);
        // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
        for (uint32_t wt = 0; wt < Wt; wt++) {
            acquire_dst();
            // copy in inputs from input_cb_index - TODO: figure out how to optimize this out
            cb_reserve_back(input_transposed_cb_index, 1);
            copy_tile(input_cb_index, wt, 0);
            // pack value tiles into cb_intermed2
            pack_tile(0, input_transposed_cb_index);
            cb_push_back(input_transposed_cb_index, 1);
            release_dst();
        }
        cb_wait_front(input_transposed_cb_index, Wt);
        cb_pop_front(input_cb_index, Wt);

        copy_tile_to_dst_init_short_with_dt(input_cb_index, index_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            acquire_dst();
            // copy in inputs from index_cb_index
            cb_reserve_back(index_transposed_cb_index, 1);
            copy_tile(index_cb_index, wt, 0);
            // pack value tiles into cb_intermed3
            pack_tile(0, index_transposed_cb_index);
            cb_push_back(index_transposed_cb_index, 1);
            release_dst();
        }
        cb_wait_front(index_transposed_cb_index, Wt);
        cb_pop_front(index_cb_index, Wt);

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logWt iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            bool direction = false;
            cb_wait_front(input_transposed_cb_index, Wt);
            cb_wait_front(index_transposed_cb_index, Wt);
            uint32_t stride = 1 << m_iter;
            for (uint32_t left_ind = 0; left_ind < Wt - stride; left_ind += 2 << m_iter) {
                uint32_t right_ind = left_ind + stride;
                acquire_dst();

                // unpack values into dest
                copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
                copy_tile(input_transposed_cb_index, left_ind, input_dest_start);
                copy_tile(input_transposed_cb_index, right_ind, input_dest_end);

                // unpack indices into dest
                copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
                copy_tile(index_transposed_cb_index, left_ind, index_dest_start);
                copy_tile(index_transposed_cb_index, right_ind, index_dest_end);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_merge(0, m_iter, K);
                // sort within the larger 32 values
                ckernel::topk_rebuild(0, (uint32_t)direction, m_iter, K, logk, true);

                // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values for
                // topk, which was in input_dest_start
                pack_reconfig_data_format(input_transposed_cb_index);
                pack_tile<true>(input_dest_start, input_transposed_cb_index, left_ind);

                // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values for
                // topk, which was in index_dest_start
                pack_reconfig_data_format(index_transposed_cb_index);
                pack_tile<true>(index_dest_start, index_transposed_cb_index, left_ind);
                release_dst();
                direction = !direction;
            }
            cb_reserve_back(input_transposed_cb_index, Wt);
            cb_reserve_back(index_transposed_cb_index, Wt);

            cb_pop_front(input_transposed_cb_index, Wt);
            cb_pop_front(index_transposed_cb_index, Wt);

            cb_push_back(input_transposed_cb_index, Wt);
            cb_push_back(index_transposed_cb_index, Wt);
        }

        // transpose value tiles and pack into output buffer
        reconfig_data_format_srca(input_transposed_cb_index);
        transpose_wh_init_short(input_transposed_cb_index);
        pack_reconfig_data_format(input_transposed_cb_index);
        cb_wait_front(input_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(values_cb_index, 1);
            transpose_wh_tile(input_transposed_cb_index, i, 0);
            pack_tile(0, values_cb_index);
            cb_push_back(values_cb_index, 1);
            release_dst();
        }
        cb_wait_front(input_transposed_cb_index, Wt);
        cb_pop_front(input_transposed_cb_index, Wt);

        // transpose index tiles and pack into output buffer
        reconfig_data_format_srca(index_transposed_cb_index);
        transpose_wh_init_short(index_transposed_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        cb_wait_front(index_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst();
            cb_reserve_back(output_ind_cb_index, 1);
            transpose_wh_tile(index_transposed_cb_index, i, 0);
            pack_tile(0, output_ind_cb_index);
            cb_push_back(output_ind_cb_index, 1);
            release_dst();
        }
        cb_wait_front(index_transposed_cb_index, Wt);
        cb_pop_front(index_transposed_cb_index, Wt);
    }
}
}  // namespace NAMESPACE
