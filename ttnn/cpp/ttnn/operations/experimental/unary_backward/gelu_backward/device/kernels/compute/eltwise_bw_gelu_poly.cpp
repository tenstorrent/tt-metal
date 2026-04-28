// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1).

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_grad_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_input = tt::CBIndex::c_1;
    constexpr uint32_t cb_grad_in = tt::CBIndex::c_2;

    using namespace compute_kernel_lib::eltwise;

    unary_op_init_common(cb_grad_out, cb_grad_in);

    auto chain = eltwise_chain(
        CopyTile<cb_grad_out, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::WaitUpfrontPopAtEnd>{},
        GeluDerivative<Approx::Exact, Dst::D1>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        eltwise_pipeline<EltwiseOutputPolicy::Bulk, EltwiseDataFormatReconfig::NONE>(
            chain, cb_grad_in, per_core_block_size);
    }
}
