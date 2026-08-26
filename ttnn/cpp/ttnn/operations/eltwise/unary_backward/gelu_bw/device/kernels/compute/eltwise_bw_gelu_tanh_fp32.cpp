// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Same tanh-approximation GELU backward formula as eltwise_bw_gelu_tanh.cpp, restricted to
// 4 live DST tiles (0-3) instead of 6.

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
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    CircularBuffer cb_grad_out_cb(cb_grad_out);
    CircularBuffer cb_input_cb(cb_input);
    CircularBuffer cb_grad_in_cb(cb_grad_in);

    constexpr float kSqrt2 = 1.41421356237309504880f;          // sqrt(2)
    constexpr float kTwoOverSqrtPi = 1.12837916709551257390f;  // 2/sqrt(pi)
    constexpr float kBeta = kSqrt2 * kTwoOverSqrtPi * 0.5f;
    constexpr float kKappa = 0.044715f;

    // The scalar SFPU ops below take the operand as packed fp32 bits.
    constexpr auto bits = [](float f) { return __builtin_bit_cast(uint32_t, f); };
    constexpr uint32_t kKappaBits = bits(kKappa);
    constexpr uint32_t kThreeKappaBits = bits(kKappa * 3.0f);
    constexpr uint32_t kBetaBits = bits(kBeta);
    constexpr uint32_t kHalfBetaBits = bits(kBeta / 2.0f);
    constexpr uint32_t kOneBits = bits(1.0f);
    constexpr uint32_t kHalfBits = bits(0.5f);

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

        copy_tile(cb_input, 0, 1);
        copy_tile(cb_input, 0, 2);  // tile[2] = x

        // tile[1] = x^3
        square_tile(1);
        mul_binary_tile(1, 2, 1);

        // Scalar immediates below replace "fill a whole tile with a constant, then do a
        // tile-by-tile binary op against it". Same fp32 arithmetic, one op instead of two.

        // tile[1] = 0.044715 * x^3
        binop_with_scalar_tile_init();
        mul_unary_tile(1, kKappaBits);

        // tile[1] = x + 0.044715 * x^3
        add_binary_tile(1, 2, 1);

        // tile[1] = sqrt(2/π) * (x + 0.044715 * x^3)
        binop_with_scalar_tile_init();
        mul_unary_tile(1, kBetaBits);

        // tile[1] = tanh(sqrt(2/π) * (x + 0.044715 * x^3))
        tanh_tile_init();
        tanh_tile(1);
        COPY_DEST_VALUES(1, 0);  // copy tanh result to tile[0]

        // CDF term: tile[1] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        binop_with_scalar_tile_init();
        add_unary_tile(1, kOneBits);
        mul_unary_tile(1, kHalfBits);

        // tile[0] = 1 - tanh^2
        square_tile(0);
        rsub_tile_init();
        rsub_tile(0, kOneBits);

        // tile[2] = (1 + 0.134145 * x**2)
        square_tile(2);  // x^2
        binop_with_scalar_tile_init();
        mul_unary_tile(2, kThreeKappaBits);
        add_unary_tile(2, kOneBits);

        // PDF term: tile[2] = 0.5 * sqrt(2/π) * (1 + 0.134145 * x^2) * (1 - tanh^2)
        mul_binary_tile(2, 0, 2);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, kHalfBetaBits);

        // tile[0] is free now (tanh/sech² no longer needed): load grad_out.
        copy_tile(cb_grad_out, 0, 0);

        // tile[2] = x * pdf term. Re-read x from the CB
        copy_tile(cb_input, 0, 3);
        mul_binary_tile(2, 3, 2);

        // result: tile[1] = cdf_term + x * pdf_term
        add_binary_tile(1, 2, 1);
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
