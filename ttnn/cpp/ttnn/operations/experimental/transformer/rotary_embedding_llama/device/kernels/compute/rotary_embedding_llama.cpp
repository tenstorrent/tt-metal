// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/matmul.h"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(8); // Index correctly in the for loop
    constexpr uint32_t num_sin_cos_rows_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t sin_cos_cb_size_in_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t Wt = get_compile_time_arg_val(11);

    mm_init();
    binary_op_init_common(rotated_in_interm_cb, cos_cb); // General Init for all binary ops

    // Get the trans_mat
    cb_wait_front(trans_mat_cb, onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    #if RELOAD_IMPL == 0
    cb_wait_front(sin_cb, sin_cos_cb_size_in_tiles);
    cb_wait_front(cos_cb, sin_cos_cb_size_in_tiles);
    #endif

    uint32_t sin_cos_row_cnt = 0;

    for (uint32_t i = 0; i < num_rows_per_core; ++i) {
        // input cb wait and reserve
        cb_wait_front(in_cb, Wt);
        #if RELOAD_IMPL == 1
        cb_wait_front(sin_cb, Wt);
        cb_wait_front(cos_cb, Wt);
        #endif

        cb_reserve_back(rotated_in_interm_cb, Wt);
        cb_reserve_back(sin_interm_cb, Wt);
        cb_reserve_back(cos_interm_cb, Wt);
        cb_reserve_back(out_cb, Wt);

        // rotated = x @ trans_mat
        mm_init_short(in_cb, trans_mat_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j, false);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        REL();
        cb_push_back(rotated_in_interm_cb, Wt);
        cb_wait_front(rotated_in_interm_cb, Wt);

        mul_tiles_init();
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // sin_interim = rotated * sin
            mul_tiles(rotated_in_interm_cb, sin_cb, j, j + (sin_cos_row_cnt * Wt), j);
            pack_tile(j, sin_interm_cb, j);
        }
        REL();
        cb_push_back(sin_interm_cb, Wt);
        cb_pop_front(rotated_in_interm_cb, Wt);

        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // cos_interim = x * cos
            mul_tiles(in_cb, cos_cb, j, j + (sin_cos_row_cnt * Wt), j);
            pack_tile(j, cos_interm_cb, j);
        }
        REL();
        cb_push_back(cos_interm_cb, Wt);
        cb_pop_front(in_cb, Wt); // Done with input
        #if RELOAD_IMPL == 1
        cb_pop_front(sin_cb, Wt);
        cb_pop_front(cos_cb, Wt);
        #endif

        cb_wait_front(cos_interm_cb, Wt);
        cb_wait_front(sin_interm_cb, Wt);
        add_tiles_init();
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // out = cos_interim + sin_interim
            add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
            pack_tile(j, out_cb, j);
        }
        REL();
        cb_push_back(out_cb, Wt);
        cb_pop_front(cos_interm_cb, Wt);
        cb_pop_front(sin_interm_cb, Wt);

        #if RELOAD_IMPL == 0
        // no-reload needs to increment this counter
        // Used a sin/cos row
        sin_cos_row_cnt++;
        // Loop back to the beginning of the sin/cos rows
        if (sin_cos_row_cnt == num_sin_cos_rows_per_core) {
            sin_cos_row_cnt = 0;
        }
        #endif
    }

    #if RELOAD_IMPL == 0
    cb_pop_front(sin_cb, sin_cos_cb_size_in_tiles);
    cb_pop_front(cos_cb, sin_cos_cb_size_in_tiles);
    #endif


    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
}
} // NAMESPACE
