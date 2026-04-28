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
    // y = x * hardsigmoid(x). FP32 path uses two CopyTiles + SFPU mul.
    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
        Hardsigmoid<Dst::D0>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
#endif
#ifdef INP_FLOAT
    // bf16: keep the FPU mul. CopyTile waits without popping; DestReuseMul
    // reuses the same waited tile (NoWaitPop) and pops it after the multiply.
    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        Hardsigmoid<Dst::D0>{},
        DestReuseMul<cb_input, Dst::D0, DestReuseInputPolicy::NoWaitPop>{});
    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
#endif
}
