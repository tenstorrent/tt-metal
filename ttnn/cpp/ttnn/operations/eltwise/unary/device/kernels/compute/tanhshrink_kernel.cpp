// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib::eltwise;

    init_sfpu(cb_input, cb_output);

#ifdef INP_FLOAT32
    // y = x - tanh(x). FP32: D0 holds x, D1 holds tanh(x), SFPU sub.
    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Tanh<Approx::Exact, Dst::D1>{},
        SfpuSub<Dst::D0, Dst::D1, Dst::D0>{});
    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
#endif
#ifdef INP_FLOAT
    // bf16: D0 = tanh(x), then DestReuseSub<ToSrcB> with cb_input.
    // ToSrcB sends DEST→srcB and CB→srcA, so result = srcA - srcB = x - tanh(x).
    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        Tanh<Approx::Exact, Dst::D0>{},
        DestReuseSub<cb_input, Dst::D0, DestReuseInputPolicy::NoWaitPop>{});
    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
#endif
}
