// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Precision-matrix workload: out = A * B (FPU multiply).
//
// FPU multiply is the classic math_fidelity-sensitive op (LoFi truncates source mantissa bits), and
// it round-trips through every CB/DEST format, so it exercises the helper's precision surface:
// input dtype (bf16 / fp32 / bfloat8_b), output dtype, fp32_dest_acc, and math_fidelity. The kernel
// is dtype-agnostic — the chain reconfigs to whatever format the CBs carry — so the whole matrix is
// driven host-side via CB dtypes + ComputeConfigDescriptor.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        BinaryFpu<
            cb_a,
            cb_b,
            BinaryFpuOp::Mul,
            BroadcastDim::None,
            InputLifecycle::Streaming,
            InputLifecycle::Streaming,
            BinaryDataFormatReconfig::Input>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output>{});
}
