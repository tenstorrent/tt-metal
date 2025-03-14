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
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "eltwise_bw_gelu_common.hpp"

#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */
#define M_SQRT1_2 0.70710678118654752440f  /* 1/sqrt(2) */

namespace NAMESPACE {

void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    // constexpr uint32_t bits_1p0 = 0x3f800000;          // 1.0f
    // constexpr uint32_t bits_0p5 = 0x3f000000;          // 0.5f
    // constexpr uint32_t bits_inv_sqrt2 = 0x3f3504f3;    // ≈ 0.70710677 (1 / sqrt(2))
    // constexpr uint32_t bits_inv_sqrt2pi = 0x3ecc422a;  // ≈ 0.3989423  (1 / sqrt(2π))
    // constexpr uint32_t bits_neg_0p5 = 0xbf000000;      // -0.5f
    constexpr float kAlpha = M_SQRT1_2;
    constexpr float kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5;

    unary_op_init_common(cb_grad_out, cb_grad_in);
    erf_tile_init();
    exp_tile_init();
    add_binary_tile_init();
    mul_binary_tile_init();
    square_tile_init();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_grad_out, per_core_block_size);
        cb_wait_front(cb_input, per_core_block_size);
        cb_reserve_back(cb_grad_in, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            tile_regs_acquire();
            tile_regs_wait();

            copy_tile(cb_grad_out, i, 0);  // grad => tile 0
            copy_tile(cb_input, i, 1);     // x => tile 1
            copy_tile(cb_input, i, 2);     // x => tile 2
            copy_tile(cb_input, i, 4);     // x => tile 7
            // Save x into another register for later
            //   tile[2] = x
            // copy_value(2, 1);
            // copy_value(7, 1);

            // Step 1: erf(x / sqrt(2))
            load_immediate_value(3, kAlpha);
            mul_binary_tile(1, 3);  // tile[1] = x / sqrt(2)
            erf_tile(1);            // tile[1] = erf( x / sqrt(2) )

            // cdf_term = 0.5 * (1.0 + erf(x / sqrt(2)))
            load_immediate_value(3, 1.0f);
            add_binary_tile(1, 3);  // tile[1] += 1.0
            load_immediate_value(3, 0.5f);
            mul_binary_tile(1, 3);  // tile[1] *= 0.5

            // Now tile[1] holds cdf_term

            // Step 2: pdf_term = x * (1 / sqrt(2π)) * exp(-x^2 / 2)
            //   tile[2] will hold exp(- x^2 / 2)
            square_tile(2);  // tile[2] = x^2
            load_immediate_value(3, -0.5f);
            mul_binary_tile(2, 3);  // tile[2] = -0.5 * x^2
            exp_tile(2);            // tile[2] = exp(- x^2 / 2)

            // multiply by (1 / sqrt(2π))
            load_immediate_value(3, kBeta);
            mul_binary_tile(2, 3);  // tile[2] *= 1 / sqrt(2π)

            // multiply by x
            mul_binary_tile(2, 4);  // tile[2] *= x

            // tile[2] now has pdf_term

            // Step 3: cdf_term + pdf_term
            add_binary_tile(1, 2);

            // Step 4: multiply by grad => tile[0]
            //   tile[0] = grad
            //   tile[1] = cdf_term + pdf_term
            mul_binary_tile(0, 1);
            pack_tile(0, cb_grad_in);

            // // store to output
            // pack_tile(0, cb_grad_in);

            tile_regs_commit();
            tile_regs_release();
        }

        cb_pop_front(cb_grad_out, per_core_block_size);
        cb_pop_front(cb_input, per_core_block_size);
        cb_push_back(cb_grad_in, per_core_block_size);
    }
}
}  // namespace NAMESPACE
