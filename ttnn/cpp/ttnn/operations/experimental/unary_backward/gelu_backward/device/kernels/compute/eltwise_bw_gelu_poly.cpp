// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 54)

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"

namespace NAMESPACE {

void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    unary_op_init_common(cb_grad_out, cb_grad_in);
    gelu_derivative_tile_init<false>();
    mul_binary_tile_init();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_reserve_back(cb_grad_in, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            cb_wait_front(cb_grad_out, per_core_block_size);
            cb_wait_front(cb_input, per_core_block_size);

            tile_regs_acquire();
            tile_regs_wait();

            copy_tile(cb_grad_out, i, 0);  // grad => tile 0
            copy_tile(cb_input, i, 1);     // x => tile 1

            // Compute GELU'(x) using polynomial approximation
            gelu_derivative_tile<false>(1);  // tile[1] = GELU'(x)

            // Multiply: grad_in = grad_out * GELU'(x)
            mul_binary_tile(0, 1, 0);  // tile[0] = tile[0] * tile[1]

            pack_tile(0, cb_grad_in);

            tile_regs_commit();
            tile_regs_release();

            cb_pop_front(cb_grad_out, per_core_block_size);
            cb_pop_front(cb_input, per_core_block_size);
        }
        cb_push_back(cb_grad_in, per_core_block_size);
    }
}
}  // namespace NAMESPACE
