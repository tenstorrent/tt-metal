// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    init_sfpu(cb_input, cb_output);

    // hardswish(x) = x * hardsigmoid(x)
    // Load x to D0 and D1, apply hardsigmoid to D0, multiply: D0 = hardsigmoid(x) * x
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0>{},
        Load<cb_input, Dst::D1>{},
        Hardsigmoid<Dst::D0>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        eltwise_op<cb_output>(chain, EltwiseTileShape::flat(per_core_block_dim));
    }
}
