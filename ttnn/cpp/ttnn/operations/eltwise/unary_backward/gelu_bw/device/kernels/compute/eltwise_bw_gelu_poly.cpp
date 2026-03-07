// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"

void kernel_main() {
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
        cb_wait_front(cb_grad_out, per_core_block_size);
        cb_wait_front(cb_input, per_core_block_size);

        // Per-tile canonical pattern: acquire → compute → commit → wait → pack → release
        // Multi-tile batching in dest is not possible here because gelu_derivative_tile
        // uses additional dest registers as scratch during polynomial evaluation.
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            tile_regs_acquire();

            copy_tile(cb_grad_out, i, 0);    // dest[0] = grad_out
            copy_tile(cb_input, i, 1);       // dest[1] = input
            gelu_derivative_tile<false>(1);  // dest[1] = GELU'(input)
            mul_binary_tile(0, 1, 0);        // dest[0] = grad_out * GELU'(input)

            tile_regs_commit();
            tile_regs_wait();

            pack_tile(0, cb_grad_in);

            tile_regs_release();
        }

        cb_pop_front(cb_grad_out, per_core_block_size);
        cb_pop_front(cb_input, per_core_block_size);
        cb_push_back(cb_grad_in, per_core_block_size);
    }
}
