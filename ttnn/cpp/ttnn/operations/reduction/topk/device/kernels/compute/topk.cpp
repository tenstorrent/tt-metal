// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/pack.h"

int32_t topk_replay_init = 0;
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

FORCE_INLINE void pack_results(uint32_t cb0, uint32_t cb1, uint32_t base_offset, uint32_t num_tiles_to_write = 2) {
    pack_reconfig_data_format(cb0);
    pack_tile(base_offset, cb0);
    if (num_tiles_to_write > 1) {
        if (cb0 != cb1) {
            pack_reconfig_data_format(cb1);
        }
        pack_tile(base_offset + 1, cb1);
    }
}

FORCE_INLINE void pop_and_transpose(uint32_t cb, uint32_t base_offset, uint32_t num_tiles_to_pop = 2) {
    reconfig_data_format_srca(cb);
    transpose_wh_init_short(cb);
    transpose_wh_tile(cb, 0, base_offset + 0);
    if (num_tiles_to_pop > 1) {
        transpose_wh_tile(cb, 1, base_offset + 1);
    }
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
    constexpr uint32_t K = get_compile_time_arg_val(10);
    constexpr uint32_t largest = get_compile_time_arg_val(11);
    constexpr uint32_t sorted = get_compile_time_arg_val(12);

    constexpr uint32_t output_tiles = (K + 31) / 32;

    ckernel::topk_tile_init();
    transpose_wh_init(input_val_cb_index, output_val_cb_index);
    transpose_wh_init(input_ind_cb_index, output_ind_cb_index);

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        uint32_t ktiles_saved = 0;

        acquire_dst();
        cb_wait_front(input_val_cb_index, 2);
        cb_wait_front(input_ind_cb_index, 2);

        pop_and_transpose(input_val_cb_index, 0);
        pop_and_transpose(input_ind_cb_index, 2);

        cb_pop_front(input_val_cb_index, 2);
        cb_pop_front(input_ind_cb_index, 2);

        ckernel::topk_local_sort(0, (int)!largest, 5);

        cb_reserve_back(result_prep_val_cb_index, output_tiles);
        cb_reserve_back(result_prep_ind_cb_index, output_tiles);

        pack_results(result_prep_val_cb_index, result_prep_val_cb_index, 0, output_tiles);
        pack_results(result_prep_ind_cb_index, result_prep_ind_cb_index, 2, output_tiles);
        ktiles_saved++;
        if (output_tiles > 1) {
            ktiles_saved++;
        }

        cb_push_back(result_prep_val_cb_index, output_tiles);
        cb_push_back(result_prep_ind_cb_index, output_tiles);

        release_dst();

        for (uint32_t count = 2; count < Wt; count++) {
            // pop next input tile and transpose into intermediate buffer
            cb_wait_front(input_val_cb_index, 1);
            cb_wait_front(input_ind_cb_index, 1);

            cb_reserve_back(transposed_val_cb_index, 1);
            cb_reserve_back(transposed_ind_cb_index, 1);

            acquire_dst();

            pop_and_transpose(input_val_cb_index, 0, 1);
            pop_and_transpose(input_ind_cb_index, 1, 1);
            pack_results(transposed_val_cb_index, transposed_ind_cb_index, 0, 2);

            release_dst();

            cb_pop_front(input_val_cb_index, 1);
            cb_pop_front(input_ind_cb_index, 1);

            // store intermediate tile for insertion sort if required for k>32
            cb_push_back(transposed_val_cb_index, 1);
            cb_push_back(transposed_ind_cb_index, 1);

            // insertion sort into result prep buffer
            for (uint32_t index = 0; index < output_tiles; index++) {
                uint32_t incr = 1;
                uint32_t cb2 = transposed_val_cb_index;
                uint32_t cb3 = transposed_ind_cb_index;
                if ((index >= (ktiles_saved - 1)) && (ktiles_saved < output_tiles)) {
                    incr = output_tiles - index;
                    ktiles_saved++;
                    index = output_tiles;
                    cb2 = result_prep_val_cb_index;
                    cb3 = result_prep_ind_cb_index;
                }

                cb_wait_front(result_prep_val_cb_index, incr);
                cb_wait_front(result_prep_ind_cb_index, incr);

                cb_wait_front(transposed_val_cb_index, 1);
                cb_wait_front(transposed_ind_cb_index, 1);

                cb_reserve_back(transposed_val_cb_index, 1);
                cb_reserve_back(transposed_ind_cb_index, 1);

                acquire_dst();

                copy_tile_to_dst_init_short_with_dt(result_prep_ind_cb_index, result_prep_val_cb_index);
                copy_tile(result_prep_val_cb_index, 0, 0);

                copy_tile_to_dst_init_short_with_dt(result_prep_val_cb_index, result_prep_ind_cb_index);
                copy_tile(result_prep_ind_cb_index, 0, 2);

                copy_tile_to_dst_init_short_with_dt(transposed_ind_cb_index, transposed_val_cb_index);
                copy_tile(transposed_val_cb_index, 0, 1);

                copy_tile_to_dst_init_short_with_dt(transposed_val_cb_index, transposed_ind_cb_index);
                copy_tile(transposed_ind_cb_index, 0, 3);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_local_sort(0, (int)!largest, 5);

                cb_reserve_back(result_prep_val_cb_index, incr);
                cb_reserve_back(result_prep_ind_cb_index, incr);

                pack_results(result_prep_val_cb_index, cb2, 0, 2);
                pack_results(result_prep_ind_cb_index, cb3, 2, 2);

                cb_pop_front(result_prep_val_cb_index, incr);
                cb_pop_front(result_prep_ind_cb_index, incr);

                cb_push_back(result_prep_val_cb_index, incr);
                cb_push_back(result_prep_ind_cb_index, incr);

                release_dst();

                cb_pop_front(transposed_val_cb_index, 1);
                cb_pop_front(transposed_ind_cb_index, 1);

                cb_push_back(transposed_val_cb_index, 1);
                cb_push_back(transposed_ind_cb_index, 1);
            }

            // remove last intermediate tile
            cb_wait_front(transposed_val_cb_index, 1);
            cb_wait_front(transposed_ind_cb_index, 1);
            cb_pop_front(transposed_val_cb_index, 1);
            cb_pop_front(transposed_ind_cb_index, 1);
        }

        cb_reserve_back(result_prep_val_cb_index, output_tiles);
        cb_reserve_back(result_prep_ind_cb_index, output_tiles);
        cb_push_back(result_prep_val_cb_index, output_tiles);
        cb_push_back(result_prep_ind_cb_index, output_tiles);

        // transpose value tiles and pack into output buffer
        transpose_and_pack(result_prep_val_cb_index, output_val_cb_index, output_tiles);

        // transpose index tiles and pack into output buffer
        transpose_and_pack(result_prep_ind_cb_index, output_ind_cb_index, output_tiles);
    }
}

}  // namespace NAMESPACE
