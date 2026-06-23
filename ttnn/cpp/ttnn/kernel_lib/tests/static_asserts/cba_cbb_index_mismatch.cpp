// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Negative compile test (G2 / SA-08): when BinaryFpu reads the SAME CB for both operands, the two
// operand indices must match. The chain dedups the B-side wait/pop against A; asymmetric indices
// would under-wait. Scalar+Bulk and Block+Bulk are each individually legal, so ONLY the
// same-CB-index guard fires. MUST fail to compile with "AIndex and BIndex must match".

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
            cb_in,
            cb_in,
            BinaryFpuOp::Add,
            BroadcastDim::None,
            InputLifecycle::Bulk,
            InputLifecycle::Bulk,
            BinaryDataFormatReconfig::Input,
            Dst::D0,
            OperandKind::Scalar,
            OperandKind::Block>{},  // same CB, mismatched indices
        PackTile<cb_out>{});
}
