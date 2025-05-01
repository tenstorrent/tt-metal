// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"

namespace NAMESPACE {

FORCE_INLINE void transpose_and_pack(uint32_t input_cb_index, uint32_t dest_cb_index, uint32_t total_tiles) {
    reconfig_data_format_srca(input_cb_index);
    transpose_wh_init_short(input_cb_index);
    pack_reconfig_data_format(input_cb_index);

    cb_wait_front(input_cb_index, 2 * total_tiles);
    for (uint32_t i = 0; i < total_tiles; ++i) {
        acquire_dst();
        cb_reserve_back(dest_cb_index, 1);
        transpose_wh_tile(input_cb_index, i, 0);
        pack_tile(0, dest_cb_index);
        cb_push_back(dest_cb_index, 1);
        release_dst();
    }
    cb_pop_front(input_cb_index, 2 * total_tiles);
}

FORCE_INLINE void pack_results(uint32_t cb0, uint32_t cb1, uint32_t base_offset) {
    pack_reconfig_data_format(cb0);
    pack_tile(base_offset, cb0);
    if (cb0 != cb1) {
        pack_reconfig_data_format(cb1);
    }
    pack_tile(base_offset + 1, cb1);
}

FORCE_INLINE void read_cb_and_transpose(uint32_t cb, uint32_t base_offset, bool get_two = true) {
    reconfig_data_format_srca(cb);
    transpose_wh_init_short(cb);
    transpose_wh_tile(cb, 0, base_offset + 0);
    if (get_two) {
        transpose_wh_tile(cb, 1, base_offset + 1);
    }
}

// just to refactor because of code overlow
FORCE_INLINE void cb_wait_pop_front(uint32_t cb, uint32_t count) {
    cb_wait_front(cb, count);
    cb_pop_front(cb, count);
}

FORCE_INLINE void cb_reserve_push_back(uint32_t cb, uint32_t count) {
    cb_reserve_back(cb, count);
    cb_push_back(cb, count);
}

void MAIN {
    constexpr uint32_t input_val_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t input_ind_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t transposed_val_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t transposed_ind_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t result_prep_val_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t result_prep_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t output_val_cb_index = get_compile_time_arg_val(6);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t output_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t largest = get_compile_time_arg_val(11);

    ckernel::topk_tile_init();
    transpose_wh_init(input_val_cb_index, output_val_cb_index);
    transpose_wh_init(input_ind_cb_index, output_ind_cb_index);

    int end_phase = 5;  // The end phase of the local sort, based on topk_local_sort documentation
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        uint32_t ktiles_saved = 0;
        /*
        General explanation of the algorithm:
        Insertion sort into buffer of K sorted tiles.
        1. First, two tiles are read from input buffer, transposed and sorted.  Result is put into result_prep buffer
        which stores up to (k+31)/32 tiles, this var is output_tiles. Lets assume for this explanation that k is 128.
        Therefore number of result tiles is 4.  After first sort, 2 tiles will be added to result buffer (2 slots still
        empty).
        2. Initial count is set to 1 to acknowledge the first two tiles we consumed. ktiles_saved now equals 2 as well.
        3. Subsequently, one tile at a time is read from input buffer, transposed, and sorted into the already sorted
        buffer at result_prep using insertion sort.
        4. The insertion sort is done by merging the new tile with the result buffer, from left to right.
        5, Each iteration produces one more tile into the result_prep while keeping all tiles in result_prep sorted.
        ktiles_saved will be 3, then 4, until ktiles_saved == output_tiles.
        6. Once the number of tiles in result_prep is equal to output_tiles, 4 in this case, new tiles will no longer be
        added to result_prep, but rather just discarded.
        7. After going through all input tiles, the result_prep buffer is then transposed and packed into the output
        buffer for write-out.

        One last note about the result_prep buffer, it is double-buffer with size of 2*output_tiles which is important.
        To maintain insertion sort logic, after each new tile is sorted, we read out from one half of the buffer, and
        store results into second half of buffer.  Therefore, we must always proceed forward to the beginning of each
        half buffer after each insertion, so a new tile is sorting starting from left most tile. Even if result_prep
        buffer is not full of tiles yet, we must move front and back pointers one full rotation of output_tiles forward
        (that's why incr changes). Once result_prep buffer is full, no extra adjustment for increment is needed and incr
        will just be 1.
        */

        // all steps had to be refactored into one loop or otherwise TRISC2 runs out of space
        uint32_t input_take = 2;
        for (uint32_t count = 1; count < Wt;
             count++) {  // start from 1 since initially we will take 2 input tiles and not just 1
            // pop next input tile(s) and transpose into intermediate buffer
            cb_wait_front(input_val_cb_index, input_take);
            cb_wait_front(input_ind_cb_index, input_take);

            cb_reserve_back(transposed_val_cb_index, input_take);
            cb_reserve_back(transposed_ind_cb_index, input_take);

            acquire_dst();

            read_cb_and_transpose(input_val_cb_index, 0, (2 == input_take));
            read_cb_and_transpose(input_ind_cb_index, 2, (2 == input_take));
            pack_reconfig_data_format(transposed_val_cb_index);
            pack_tile(0, transposed_val_cb_index);
            if (input_take == 2) {
                pack_tile(1, transposed_val_cb_index);
            }

            pack_reconfig_data_format(transposed_ind_cb_index);
            pack_tile(2, transposed_ind_cb_index);
            if (input_take == 2) {
                pack_tile(3, transposed_ind_cb_index);
            }

            release_dst();

            cb_pop_front(input_val_cb_index, input_take);
            cb_pop_front(input_ind_cb_index, input_take);

            // store intermediate tile for insertion sort if required for k>32
            cb_push_back(transposed_val_cb_index, input_take);
            cb_push_back(transposed_ind_cb_index, input_take);

            // insertion sort into result prep buffer
            for (uint32_t index = 0; index < output_tiles; index++) {
                uint32_t incr = 1;
                uint32_t transposed_offset = 0;
                uint32_t cb0 = result_prep_val_cb_index;
                uint32_t cb1 = result_prep_ind_cb_index;
                uint32_t cb2 = transposed_val_cb_index;
                uint32_t cb3 = transposed_ind_cb_index;
                uint32_t in_cb_offset = incr;

                if (ktiles_saved == 0) {
                    incr = output_tiles;
                    transposed_offset = 1;
                    cb0 = transposed_val_cb_index;
                    cb1 = transposed_ind_cb_index;
                    cb2 = result_prep_val_cb_index;
                    cb3 = result_prep_ind_cb_index;
                    ktiles_saved += 2;
                    index = output_tiles;
                    in_cb_offset = input_take;
                } else if ((index >= (ktiles_saved - 1)) && (ktiles_saved < output_tiles)) {
                    incr = output_tiles - index;
                    ktiles_saved++;
                    index = output_tiles;
                    cb2 = result_prep_val_cb_index;
                    cb3 = result_prep_ind_cb_index;
                    in_cb_offset = incr;
                }

                cb_wait_front(cb0, in_cb_offset);
                cb_wait_front(cb1, in_cb_offset);
                if (transposed_offset == 0) {
                    cb_wait_front(transposed_val_cb_index, 1);
                    cb_wait_front(transposed_ind_cb_index, 1);
                }

                cb_reserve_back(transposed_val_cb_index, 1);
                cb_reserve_back(transposed_ind_cb_index, 1);

                acquire_dst();

                copy_tile_to_dst_init_short_with_dt(cb1, cb0);
                copy_tile(cb0, 0, 0);

                copy_tile_to_dst_init_short_with_dt(cb0, cb1);
                copy_tile(cb1, 0, 2);

                copy_tile_to_dst_init_short_with_dt(transposed_ind_cb_index, transposed_val_cb_index);
                copy_tile(transposed_val_cb_index, transposed_offset, 1);

                copy_tile_to_dst_init_short_with_dt(transposed_val_cb_index, transposed_ind_cb_index);
                copy_tile(transposed_ind_cb_index, transposed_offset, 3);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_local_sort(0, (int)!largest, end_phase);

                cb_reserve_back(result_prep_val_cb_index, incr);
                cb_reserve_back(result_prep_ind_cb_index, incr);

                pack_results(result_prep_val_cb_index, cb2, 0);
                pack_results(result_prep_ind_cb_index, cb3, 2);

                cb_pop_front(cb0, in_cb_offset);
                cb_pop_front(cb1, in_cb_offset);

                cb_push_back(result_prep_val_cb_index, incr);
                cb_push_back(result_prep_ind_cb_index, incr);

                release_dst();

                if (transposed_offset == 0) {
                    cb_pop_front(transposed_val_cb_index, 1);
                    cb_pop_front(transposed_ind_cb_index, 1);
                }

                cb_push_back(transposed_val_cb_index, 1);
                cb_push_back(transposed_ind_cb_index, 1);
            }

            input_take = 1;

            // remove last intermediate tile
            cb_wait_pop_front(transposed_val_cb_index, 1);
            cb_wait_pop_front(transposed_ind_cb_index, 1);
        }

        cb_reserve_push_back(result_prep_val_cb_index, output_tiles);
        cb_reserve_push_back(result_prep_ind_cb_index, output_tiles);

        // transpose value tiles and pack into output buffer
        transpose_and_pack(result_prep_val_cb_index, output_val_cb_index, output_tiles);

        // transpose index tiles and pack into output buffer
        transpose_and_pack(result_prep_ind_cb_index, output_ind_cb_index, output_tiles);
    }
}

}  // namespace NAMESPACE
