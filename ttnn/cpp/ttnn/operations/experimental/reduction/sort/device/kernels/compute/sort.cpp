// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_binary.h"

#include "sort_common.hpp"

#include "debug/dprint.h"

namespace NAMESPACE {
/*
The sorting algorithm is based on Bitonic Merge Sort.

 The program operates on the arrangement of input data in the form of tiles. After the data passes through
 preprocessing, the dimension by which the data is to be sorted is the last dimension in the tensor, hence from the
 point of view of the arrangement of tiles, the sorting is practically based on sorting successive rows of the matrix.

 First, the program reads one full row of tiles (size Wt) from DRAM to L1 and generates a corresponding set of tiles
 containing the data indices in the initial arrangement. The basis for the described sorting implementation is
 ckernel::topk_local_sort LLK, which takes as input arguments to the DST register two tiles on which
 the sorting is to be performed and two tiles containing the indices of this data. LLK sorts the two input tiles in the
 inplace method and sets the indices of the data that have been changed in place. However, it performs sorting on
 columns, hence the additional step of performing transposition. Since LLK always takes a set of two tiles, their number
 in the Wt dimension must always be a multiple of 64 (2 * Tile_Width (32)).

 sort_Wt_tiles_row_to_bitonic_sequence takes pairs of tiles and sorts them among themselves changing the sorting order
 (for the first pair according to the desired order, for the next one vice versa, etc.), as a result we get a set of
 sorted pairs of tiles with a variable sorting order.

 The next step is to sort the tiles among themselves, so that the entire row of input data is sorted. As in the
 bitonic merge sort algorithm, this is done in stages – the next indexes of tiles in CB are calculated, which are to be
 sorted among themselves. Finally, all tiles have their values ​​sorted with respect to one dimension and are
 ready to be transposed to the desired dimension and written to DRAM memory.

 Example:
 The input tensor is a 32x128 matrix, which translates to 1x4 tiles: T0 T1 T2 T3, we sort the values ​​ascending.
 Pairwise sorting: Tiles T0 and T1 are sorted as a pair ascending, T2 and T3 are sorted as a pair descending.
 Sorting among themselves: Stage 1 - we sort T0 and T2 ascending and T1 and T3 ascending (we do not change the order),
                           Stage 2 – We sort T0 and T1 ascending and T2 and T3 ascending.
 Data saving: Values ​​have been sorted by a common dimension and are ready to be saved.
 */
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
    constexpr bool descending = get_compile_time_arg_val(8);
    constexpr bool stable =
        get_compile_time_arg_val(9);  // TODO: In the future change LLK to have the option or add additional step with
                                      // checking values and indexes after the sorting

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
            /*switch_dir=*/true,
            ascending,
            /*end_phase(log2(K))=*/5);

        // Wait for bitonic sequence of Wt tiles
        cb_wait_front(input_tensor_transposed_cb_index, Wt);
        cb_wait_front(index_tensor_transposed_cb_index, Wt);

        // Sort and merge step of bitonic merge sort
        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }

        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        // Determine direction for this comparison block
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir = ascending_block == ascending;

                        // Get indexes of tiles to compare
                        const uint32_t left_tile_id = i;
                        const uint32_t right_tile_id = j;

                        acquire_dst();

                        copy_tile_to_dst_init_short_with_dt(
                            index_tensor_transposed_cb_index, input_tensor_transposed_cb_index);
                        copy_tile(input_tensor_transposed_cb_index, left_tile_id, input_dest_start);
                        copy_tile(input_tensor_transposed_cb_index, right_tile_id, input_dest_end);

                        copy_tile_to_dst_init_short_with_dt(
                            input_tensor_transposed_cb_index, index_tensor_transposed_cb_index);
                        copy_tile(index_tensor_transposed_cb_index, left_tile_id, index_dest_start);
                        copy_tile(index_tensor_transposed_cb_index, right_tile_id, index_dest_end);

                        ckernel::topk_local_sort(0, (int)dir, 5);

                        pack_reconfig_data_format(input_tensor_transposed_cb_index);
                        pack_tile<true>(input_dest_start, input_tensor_transposed_cb_index, left_tile_id);
                        pack_tile<true>(input_dest_end, input_tensor_transposed_cb_index, right_tile_id);

                        pack_reconfig_data_format(index_tensor_transposed_cb_index);
                        pack_tile<true>(index_dest_start, index_tensor_transposed_cb_index, left_tile_id);
                        pack_tile<true>(index_dest_end, index_tensor_transposed_cb_index, right_tile_id);

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
