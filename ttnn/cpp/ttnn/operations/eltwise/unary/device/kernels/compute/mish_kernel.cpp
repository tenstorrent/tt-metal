// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const bool use_approx = (get_arg_val<uint32_t>(0) != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);

    using namespace compute_kernel_lib;

    init_sfpu(cb_input, cb_output);

    // mish(x) = x * tanh(log1p(exp(x)))
    // Load x to D0 (compute path) and D1 (preserve x for final mul).
    // D0: exp → log1p → tanh → result in D0.
    // Final: D0 = tanh(softplus(x)) * x via SfpuMul(D0, D1).
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        if (use_approx) {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0>{},
                Load<cb_input, Dst::D1>{},
                Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Fast, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
            eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(
                chain, EltwiseTileShape::flat(per_core_block_dim));
        } else {
            auto chain = sfpu_chain(
                Load<cb_input, Dst::D0>{},
                Load<cb_input, Dst::D1>{},
                Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
                Log1p<Approx::Exact, Dst::D0>{},
                Tanh<Approx::Exact, Dst::D0>{},
                SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
            eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(
                chain, EltwiseTileShape::flat(per_core_block_dim));
        }
    }
}
