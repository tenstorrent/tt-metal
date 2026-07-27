// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DestReuseBinary element coverage: out = (A + B) * C, reusing the DEST result as an FPU operand.
//   D1 = A + B (BinaryFpu) ; D1 = DEST(D1) * C (DestReuseBinary Mul, DEST_TO_SRCA) ; pack D1
// DestReuseBinary feeds DEST back into the FPU instead of a second CB read. The (A+B)*C golden fails
// if the DEST operand is mis-routed. Using D1 also proves that dest reuse writes back to its input slot.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        BinaryFpu<input(cb_a), input(cb_b), BinaryFpuOp::Add, BroadcastDim::None, Dst::D1>{},
        DestReuseBinary<input(cb_c), BinaryFpuOp::Mul, DestReuseType::DEST_TO_SRCA, Dst::D1>{},
        PackTile<output(cb_out), Dst::D1>{});
}
