// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    const uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);

    init_sfpu(cb_input, cb_output);

    // tanhshrink(x) = x - tanh(x)
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
        Tanh<Approx::Exact, Dst::D1>{},
        SfpuSub<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(chain, EltwiseTileShape::flat(per_core_block_dim));
    }
}
