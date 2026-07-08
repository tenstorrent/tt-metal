// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DestReuseBinary element coverage (untested element type).
//
// out = (A + B) * C, computed by reusing the DEST result as one operand of the next op:
//   D0 = A + B                                         (BinaryFpu)
//   D0 = DEST(D0) * C   via DEST_TO_SRCA: CB C -> srcb, DEST -> srca   (DestReuseBinary, Mul)
//   pack D0
// DestReuseBinary feeds DEST back into the FPU instead of a second CB read — a distinct element with
// its own reconfig/side selection. The (A+B)*C golden fails if the DEST operand is mis-routed.

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
        BinaryFpu<
            cb_a,
            cb_b,
            BinaryFpuOp::Add,
            BroadcastDim::None,
            InputLifecycle::Streaming,
            InputLifecycle::Streaming,
            BinaryDataFormatReconfig::Input,
            Dst::D0>{},
        DestReuseBinary<
            cb_c,
            BinaryFpuOp::Mul,
            DestReuseType::DEST_TO_SRCA,
            InputLifecycle::Streaming,
            DestReuseReconfig::Input,
            Dst::D0,
            Dst::D0>{},
        PackTile<cb_out, OutputLifecycle::Streaming, PackTileReconfig::Output, Dst::D0>{});
}
