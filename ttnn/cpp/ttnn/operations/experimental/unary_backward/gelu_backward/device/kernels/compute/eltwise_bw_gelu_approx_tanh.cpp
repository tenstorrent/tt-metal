// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/circular_buffer.h"

#define M_SQRT2 1.41421356237309504880f    /* sqrt(2) */
#define M_2_SQRTPI 1.12837916709551257390f /* 2/sqrt(pi) */

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    CircularBuffer cb_grad_out_cb(cb_grad_out);
    CircularBuffer cb_input_cb(cb_input);
    CircularBuffer cb_grad_in_cb(cb_grad_in);

    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
    constexpr float kKappa = 0.044715;

    unary_op_init_common(cb_grad_out, cb_grad_in);
    add_binary_tile_init();
    mul_binary_tile_init();
    square_tile_init();
    tanh_tile_init();
    sub_binary_tile_init();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_grad_in_cb.reserve_back(1);
        cb_grad_out_cb.wait_front(1);
        cb_input_cb.wait_front(1);

        tile_regs_acquire();

        copy_tile(cb_grad_out, 0, 0);
        copy_tile(cb_input, 0, 1);
        copy_tile(cb_input, 0, 2);  // tile[2] = x
        copy_tile(cb_input, 0, 5);  // tile[5] = x

        // tile[1] = x^3
        square_tile(1);
        mul_binary_tile(1, 2, 1);

        // tile[1] = 0.044715 * x^3
        fill_tile(3, kKappa);
        mul_binary_tile(1, 3, 1);

        // tile[1] = x + 0.044715 * x^3
        add_binary_tile(1, 2, 1);

        // tile[1] = sqrt(2/π) * (x + 0.044715 * x^3)
        fill_tile(3, kBeta);
        mul_binary_tile(1, 3, 1);

        // tile[1] = tanh(sqrt(2/π) * (x + 0.044715 * x^3))
        tanh_tile_init();
        tanh_tile(1);
        copy_dest_values(1, 4);  // save tanh to tile[4]

        // CDF term: tile[1] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        fill_tile(3, 1.0f);
        add_binary_tile(1, 3, 1);
        fill_tile(3, 0.5f);
        mul_binary_tile(1, 3, 1);

        // tile[4] = 1 - tanh^2
        square_tile(4);
        fill_tile(3, 1.0f);
        sub_binary_tile(3, 4, 3);
        copy_dest_values(3, 4);

        // tile[2] = (1 + 0.134145 * x**2)
        fill_tile(3, kKappa * 3.0f);
        square_tile(2);            // x^2
        mul_binary_tile(2, 3, 2);  // 0.134145 * x**2
        fill_tile(3, 1.0f);
        add_binary_tile(2, 3, 2);  // 1 + 0.134145 * x**2

        // PDF term: tile[2] = 0.5 * sqrt(2/π) * (1 + 0.134145 * x^2) * (1 - tanh^2)
        mul_binary_tile(2, 4, 2);
        fill_tile(3, kBeta / 2.0f);
        mul_binary_tile(2, 3, 2);

        // tile[2] = x * pdf term
        copy_dest_values(5, 3);
        mul_binary_tile(2, 3, 2);

        // result: tile[1] = grad * (cdf_term + x * pdf_term)
        add_binary_tile(1, 2, 1);  // cdf_term + x * pdf_term
        // tile[0] = grad * (cdf_term + x * pdf_term)
        mul_binary_tile(0, 1, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_grad_in);

        tile_regs_release();

        cb_grad_out_cb.pop_front(1);
        cb_input_cb.pop_front(1);
        cb_grad_in_cb.push_back(1);
    }
}
