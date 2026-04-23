// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    const uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    // grad_in = grad_out * GELU'(input)
    // Two-CB chain: both inputs use WaitAndPop (per-tile wait/pop).
    // Both CBs must have the same data format (standard for backward kernels).
    unary_op_init_common(cb_grad_out, cb_grad_in);

    auto chain = sfpu_chain(
        Load<cb_grad_out, Dst::D0>{},           // D0 = grad_out
        Load<cb_input, Dst::D1>{},              // D1 = input
        GeluDerivative<false, Dst::D1>{},       // D1 = GELU'(input)
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});  // D0 = grad_out * GELU'(input)

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        eltwise_op<cb_grad_in, Dst::D0, EltwiseOutputPolicy::Bulk>(chain, EltwiseTileShape::flat(per_core_block_size));
    }
}
