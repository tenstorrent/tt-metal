// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Init-hoisting equality — PER-TILE side: N separate eltwise_chain(single()) calls, each re-emitting
// its own init (Exp re-initialized N times). Paired with hoist_single_call.cpp (init hoisted once);
// for identical input the two outputs must be bit-identical.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    for (uint32_t i = 0; i < n; ++i) {
        eltwise_chain(EltwiseShape::single(), CopyTile<input(cb_in)>{}, Exp<>{}, PackTile<output(cb_out)>{});
    }
}
