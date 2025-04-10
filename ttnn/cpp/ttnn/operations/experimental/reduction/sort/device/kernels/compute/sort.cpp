// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "sort_common.hpp"

#include "debug/dprint.h"

// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;
namespace NAMESPACE {

void MAIN {
    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t logWt = get_compile_time_arg_val(8);
    constexpr bool descending = get_compile_time_arg_val(9);
    constexpr bool stable = get_compile_time_arg_val(10);

    constexpr uint32_t one_tile = 1;

    constexpr uint32_t input_dest_start = 0;
    constexpr uint32_t index_dest_start = 2;
    constexpr uint32_t input_dest_end = 1;
    constexpr uint32_t index_dest_end = 3;

    ckernel::topk_tile_init();
    transpose_wh_init(input_tensor_cb_index, input_tensor_transposed_cb_index);

    for (uint32_t h = 0; h < Ht; h++) {
        const bool ascending = !descending;

        sort_Wt_tiles_row_to_bitonic_sequence(
            input_tensor_cb_index,
            index_tensor_cb_index,
            input_tensor_transposed_cb_index,
            index_tensor_transposed_cb_index,
            Wt,
            true,
            ascending,
            5);

        // Wait for bitonic sequence of Wt tiles
        cb_wait_front(input_tensor_transposed_cb_index, Wt);
        cb_wait_front(index_tensor_transposed_cb_index, Wt);

        // Sort and merge step of bitonic merge sort
        uint32_t stages = 0;
        for (uint32_t temp = Wt; temp > 1; temp >>= 1) {
            stages++;
        }

        for (uint32_t stage = 2; stage <= stages; stage++) {  // TODO: Check if the first step is necessary
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        const uint32_t left_tile_id = i;
                        const uint32_t right_tile_id = j;
                        DPRINT_MATH(
                            DPRINT << "INPUT MATRIX ROW: " << U32(h) << " SORTING TILES: " << U32(left_tile_id)
                                   << " and " << U32(right_tile_id) << ENDL());

                        acquire_dst();

                        copy_tile_to_dst_init_short_with_dt(
                            index_tensor_transposed_cb_index, input_tensor_transposed_cb_index);
                        copy_tile(input_tensor_transposed_cb_index, left_tile_id, 0);
                        copy_tile(input_tensor_transposed_cb_index, right_tile_id, 1);

                        copy_tile_to_dst_init_short_with_dt(
                            input_tensor_transposed_cb_index, index_tensor_transposed_cb_index);
                        copy_tile(index_tensor_transposed_cb_index, left_tile_id, 2);
                        copy_tile(index_tensor_transposed_cb_index, right_tile_id, 3);

                        ckernel::topk_local_sort(0, (int)ascending, 5);

                        pack_reconfig_data_format(input_tensor_transposed_cb_index);
                        pack_tile<true>(0, input_tensor_transposed_cb_index, left_tile_id);
                        pack_tile<true>(1, input_tensor_transposed_cb_index, right_tile_id);

                        pack_reconfig_data_format(index_tensor_transposed_cb_index);
                        pack_tile<true>(2, index_tensor_transposed_cb_index, left_tile_id);
                        pack_tile<true>(3, index_tensor_transposed_cb_index, right_tile_id);

                        release_dst();
                    }
                }
            }
        }

        cb_reserve_back(input_tensor_transposed_cb_index, Wt);
        cb_reserve_back(index_tensor_transposed_cb_index, Wt);

        cb_pop_front(input_tensor_transposed_cb_index, Wt);
        cb_pop_front(index_tensor_transposed_cb_index, Wt);

        cb_push_back(input_tensor_transposed_cb_index, Wt);
        cb_push_back(index_tensor_transposed_cb_index, Wt);

        // Values tensor
        transpose_and_pack(input_tensor_transposed_cb_index, value_tensor_cb_index, Wt);

        // Indexes tensor
        transpose_and_pack(index_tensor_transposed_cb_index, index_tensor_output_cb_index, Wt);
    }  // Ht loop
}

}  // namespace NAMESPACE
