// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: BinaryFpu reading the SAME CB for both operands must use the same input
// wait/pop pair because the chain dedups the B-side wait/pop against A. (Upfront, AtEnd) and
// (Upfront, None) are each legal with Scalar, so only the same-CB-policy guard fires.
// MUST fail to compile with "same wait/pop policies".

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        BinaryFpu<
            input(cb_in, WaitPolicy::Upfront, PopPolicy::AtEnd),
            input(cb_in, WaitPolicy::Upfront, PopPolicy::None),
            BinaryFpuOp::Add,
            BroadcastDim::None>{},
        PackTile<output(cb_out)>{});
}
