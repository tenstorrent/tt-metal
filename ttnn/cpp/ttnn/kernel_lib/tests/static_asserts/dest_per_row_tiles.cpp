// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

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
            input(cb_a, InputLifecycle::Bulk, OperandKind::Block),
            input(cb_b, InputLifecycle::Bulk, OperandKind::Block),
            BinaryFpuOp::Add,
            BroadcastDim::None,
            Dst::D0,
            DestAccumulation::PerRow>{},
        PackTile<output(
            cb_out,
            OutputLifecycle::DestAccumulation,
            DataFormatReconfig::Enabled,
            PackRelu::Disabled,
            L1Accumulation::Disabled,
            DestAccumulation::PerRow)>{});
}
