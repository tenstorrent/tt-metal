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

#include "topk_common_funcs.hpp"

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
    constexpr uint32_t largest = get_compile_time_arg_val(12);
    constexpr uint32_t sorted = get_compile_time_arg_val(13);

    // dest indices for where to unpack the tiles for the llk
    // the input goes in index 0,1 and the index goes in index 2,3
    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    constexpr uint32_t tiles_per_seq = (K + 31) / 32;
    bool switch_dir = (K == 64);
    int seq_per_2tiles = std::max((2 * 32) / K, (uint32_t)2);

    // init pack, compute and unpack
    init_sfpu(input_cb_index, tt::CBIndex::c_16);
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
            release_dst();
        }
        cb_push_back(input_transposed_cb_index, Wt);
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

        uint32_t num_k_sequences = (Wt * 32) / K;

        // iterative divide and conquer on pairs of tiles (bitonic topk merge and rebuild)
        // first iteration we compare 0th and 1st tile, then 2nd and 3rd, etc. We get the sorted top 32 values in each
        // pair. second iteration we compare 0th and 2nd tile, then 4th and 6th, etc. logWt iteration we compare 0th and
        // Wt/2 tile single buffer as we can pack tiles back in-place
        for (uint32_t m_iter = 0; m_iter < logWt; ++m_iter) {
            process_iteration(
                m_iter,
                K,
                Wt,
                num_k_sequences,
                tiles_per_seq,
                input_transposed_cb_index,
                index_transposed_cb_index,
                input_dest_start,
                input_dest_end,
                index_dest_start,
                index_dest_end,
                largest,
                switch_dir,
                logk,
                seq_per_2tiles,
                largest);
        }

        // transpose value tiles and pack into output buffer
        transpose_and_pack(input_transposed_cb_index, values_cb_index, Kt, Wt);

        // transpose index tiles and pack into output buffer
        transpose_and_pack(index_transposed_cb_index, output_ind_cb_index, Kt, Wt);
    }
}
}  // namespace NAMESPACE
