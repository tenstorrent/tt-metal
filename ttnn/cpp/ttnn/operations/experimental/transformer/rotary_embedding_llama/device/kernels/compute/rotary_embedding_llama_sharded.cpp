// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Ht = get_compile_time_arg_val(9);  // How many rows (tiles) in n_heads dimension

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    // Get the trans_mat
    cb_reserve_back(trans_mat_cb, onetile);
    cb_push_back(trans_mat_cb, onetile);
    cb_wait_front(trans_mat_cb, onetile);

    // Get the sin/cos matrices
    // TODO: To parallelize across multiple batch, this should be in a batch loop
    cb_reserve_back(sin_cb, Wt);
    cb_reserve_back(cos_cb, Wt);

    cb_push_back(sin_cb, Wt);
    cb_push_back(cos_cb, Wt);

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        cb_reserve_back(rotated_in_interm_cb, Wt);

        // Get the input
        cb_reserve_back(in_cb, Wt);
        cb_push_back(in_cb, Wt);
        cb_wait_front(in_cb, Wt);

        // Do the computation

        // rotated = x @ trans_mat
        mm_init_short(in_cb, trans_mat_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, trans_mat_cb, j, 0, j);
            pack_tile(j, rotated_in_interm_cb, j);
        }
        REL();
        cb_push_back(rotated_in_interm_cb, Wt);

        // sin_interim = rotated * sin (ROW bcast; sin_cb pre-waited globally, NoWaitNoPop)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            rotated_in_interm_cb, sin_cb, sin_interm_cb, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // cos_interim = x * cos (ROW bcast; cos_cb pre-waited globally, NoWaitNoPop)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            in_cb, cos_cb, cos_interm_cb, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // out = cos_interim + sin_interim
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cos_interm_cb, sin_interm_cb, out_cb, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
    }

    // Done with the sin/cos matrices, so remove from CB
    cb_pop_front(sin_cb, Wt);
    cb_pop_front(cos_cb, Wt);

    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
}
