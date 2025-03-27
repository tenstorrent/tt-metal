// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    // TODO: Add back early return? Currently, running out of code size in TRISC2 by 4B
    // const bool has_work = get_arg_val<uint32_t>(0);
    // if (!has_work) {
    //     return;
    // }
    const bool is_q = get_arg_val<uint32_t>(0);

    // First 6 args for q and k heads
    // - First 3 are for q
    // - Next 3 are for k
    constexpr uint32_t q_in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t q_Ht = get_compile_time_arg_val(2);
    constexpr uint32_t k_in_cb = get_compile_time_arg_val(3);
    constexpr uint32_t k_out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t k_Ht = get_compile_time_arg_val(5);
    uint32_t in_cb = q_in_cb;
    uint32_t out_cb = q_out_cb;
    uint32_t Ht = q_Ht;
    if (!is_q) {
        in_cb = k_in_cb;
        out_cb = k_out_cb;
        Ht = k_Ht;
    }

    constexpr uint32_t Wt = get_compile_time_arg_val(6);  // How many rows (tiles) in n_heads dimension

    constexpr uint32_t cos_cb = get_compile_time_arg_val(7);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(8);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(9);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(10);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(11);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(12);

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    /* Unnecessary CB APIs (comment out for code size)
    // Get the trans_mat
    constexpr uint32_t onetile = 1;
    cb_reserve_back(trans_mat_cb, onetile);
    cb_push_back(trans_mat_cb, onetile);
    cb_wait_front(trans_mat_cb, onetile);

    // Get the sin/cos matrices
    // TODO: To parallelize across multiple batch, this should be in a batch loop
    cb_reserve_back(sin_cb, Wt);
    cb_reserve_back(cos_cb, Wt);

    cb_push_back(sin_cb, Wt);
    cb_push_back(cos_cb, Wt);
    */

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        cb_reserve_back(rotated_in_interm_cb, Wt);
        cb_reserve_back(sin_interm_cb, Wt);
        cb_reserve_back(cos_interm_cb, Wt);
        cb_reserve_back(out_cb, Wt);

        // Get the input
        cb_reserve_back(in_cb, Wt);
        cb_push_back(in_cb, Wt);
        cb_wait_front(in_cb, Wt);

        // Do the computation

        // rotated = x @ trans_mat
        mm_init_short(in_cb, trans_mat_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, trans_mat_cb, j, 0, j, false);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        REL();
        cb_push_back(rotated_in_interm_cb, Wt);
        cb_wait_front(rotated_in_interm_cb, Wt);

        mul_bcast_rows_init_short(rotated_in_interm_cb, sin_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // sin_interim = rotated * sin
            mul_tiles_bcast<BroadcastType::ROW>(rotated_in_interm_cb, sin_cb, j, j, j);
            pack_tile(j, sin_interm_cb, j);
        }
        REL();
        cb_push_back(sin_interm_cb, Wt);
        cb_pop_front(rotated_in_interm_cb, Wt);

        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // cos_interim = x * cos
            mul_tiles_bcast<BroadcastType::ROW>(in_cb, cos_cb, j, j, j);
            pack_tile(j, cos_interm_cb, j);
        }
        REL();
        cb_push_back(cos_interm_cb, Wt);
        cb_pop_front(in_cb, Wt);  // Done with input

        cb_wait_front(sin_interm_cb, Wt);
        cb_wait_front(cos_interm_cb, Wt);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // out = cos_interim + sin_interim
            add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
            pack_tile(j, out_cb, j);
        }
        REL();
        cb_push_back(out_cb, Wt);
        cb_pop_front(sin_interm_cb, Wt);
        cb_pop_front(cos_interm_cb, Wt);
    }

    /* Unnecessary CB APIs (comment out for code size)
    // Done with the sin/cos matrices, so remove from CB
    cb_pop_front(sin_cb, Wt);
    cb_pop_front(cos_cb, Wt);

    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
    */
}
}  // namespace NAMESPACE
