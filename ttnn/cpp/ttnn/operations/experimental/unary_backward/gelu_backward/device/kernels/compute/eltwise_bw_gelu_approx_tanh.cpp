// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

#define M_SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kKappa = 0.044715;

    unary_op_init_common(cb_grad_out, cb_grad_in);
    add_binary_tile_init();
    mul_binary_tile_init();
    square_tile_init();
    tanh_tile_init();
    sub_binary_tile_init();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_reserve_back(cb_grad_in, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            cb_wait_front(cb_grad_out, per_core_block_size);
            cb_wait_front(cb_input, per_core_block_size);

            tile_regs_acquire();
            tile_regs_wait();
            copy_tile(cb_grad_out, i, 0);
            copy_tile(cb_input, i, 1);
            copy_tile(cb_input, i, 2);  // tile[2] = x
            copy_tile(cb_input, i, 8);  // tile[8] = x

            // tile[1] = x^3
            square_tile(1);
            mul_binary_tile(1, 2);

            // tile[1] = 0.044715 * x^3
            fill_tile(3, kKappa);
            mul_binary_tile(1, 3);

            // tile[1] = x + 0.044715 * x^3
            add_binary_tile(1, 2);

            // tile[1] = sqrt(2/π) * (x + 0.044715 * x^3)
            fill_tile(3, kBeta);
            mul_binary_tile(1, 3);

            // tile[1] = tanh(sqrt(2/π) * (x + 0.044715 * x^3))
            tanh_tile_init();
            tanh_tile(1);
            copy_dest_values(4, 1);  // save tanh to tile[7]

            // CDF term: tile[1] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            fill_tile(3, 1.0f);
            add_binary_tile(1, 3);
            fill_tile(3, 0.5f);
            mul_binary_tile(1, 3);

            // tile[4] = 1 - tanh^2
            square_tile(4);
            fill_tile(3, 1.0f);
            sub_binary_tile(3, 4);
            copy_dest_values(4, 3);

            // tile[2] = (1 + 0.134145 * x**2)
            fill_tile(3, kKappa * 3.0f);
            square_tile(2);         // x^2
            mul_binary_tile(2, 3);  // 0.134145 * x**2
            fill_tile(3, 1.0f);
            add_binary_tile(2, 3);  // 1 + 0.134145 * x**2

            // PDF term: tile[2] = 0.5 * sqrt(2/π) * (1 + 0.134145 * x^2) * (1 - tanh^2)
            mul_binary_tile(2, 4);
            fill_tile(3, kBeta / 2.0f);
            mul_binary_tile(2, 3);

            // tile[2] = x * pdf tern
            copy_dest_values(3, 8);
            mul_binary_tile(2, 3);

            // result: tile[1] = grad * (cdf_term + x * pdf_term)
            add_binary_tile(1, 2);  // cdf_term + x * pdf_term
            // tile[0] = grad * (cdf_term + x * pdf_term)
            mul_binary_tile(0, 1);

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
