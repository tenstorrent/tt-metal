// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Precision-matrix workload: out = A * B (FPU multiply). Multiply is fidelity-sensitive (LoFi
// truncates source mantissa bits) and round-trips through every CB/DEST format, so it exercises the
// precision surface: input/output dtype, fp32_dest_acc, math_fidelity. The kernel is dtype-agnostic
// (the chain reconfigs to the CBs' formats), so the whole matrix is driven host-side.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(EltwiseShape::tiles(n), BinaryFpu<cb_a, cb_b, BinaryFpuOp::Mul>{}, PackTile<cb_out>{});
}
