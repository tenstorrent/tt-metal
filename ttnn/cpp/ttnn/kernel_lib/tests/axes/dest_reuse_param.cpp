// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DestReuseBinary matrix: reuse-direction x op. Stage 1 (fixed) D0 = A + B; stage 2 feeds DEST(D0)
// and CB C into the FPU per ReuseType:
//   DEST_TO_SRCA -> (A+B) op C     DEST_TO_SRCB -> C op (A+B)
// For Sub the two directions differ, so the matrix proves DEST is routed to the correct unpack lane.
//
// CT args: [n, reuse (0=SRCA,1=SRCB), op (0=Add,1=Sub,2=Mul)].

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t reuse = get_compile_time_arg_val(1);
    constexpr uint32_t op = get_compile_time_arg_val(2);

    using namespace compute_kernel_lib;
    constexpr DestReuseType R = (reuse == 0) ? DestReuseType::DEST_TO_SRCA : DestReuseType::DEST_TO_SRCB;
    constexpr BinaryFpuOp OP = (op == 0) ? BinaryFpuOp::Add : ((op == 1) ? BinaryFpuOp::Sub : BinaryFpuOp::Mul);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    eltwise_chain(
        EltwiseShape::tiles(n),
        BinaryFpu<
            cb_a,
            cb_b,
            BinaryFpuOp::Add,
            BroadcastDim::None,
            InputLifecycle::Streaming,
            InputLifecycle::Streaming,
            BinaryDataFormatReconfig::Input,
            Dst::D0>{},
        DestReuseBinary<cb_c, OP, R, InputLifecycle::Streaming, DestReuseReconfig::Input, Dst::D0, Dst::D0>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{});
}
