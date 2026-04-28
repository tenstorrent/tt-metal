// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#ifdef INP_FLOAT32
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
#endif

#ifdef INP_FLOAT
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

#ifdef INP_FLOAT32
    // mish(x) = x * tanh(log1p(exp(x)))
    // Two CopyTiles on cb_input — fan-out: D0 takes exp/log1p/tanh path,
    // D1 holds the original x for the final mul. WaitNoPop + NoWaitPop
    // means one wait + one pop per iteration, two physical copy_tile calls.
    using namespace compute_kernel_lib::eltwise;
    if (use_approx) {
        auto chain = eltwise_chain(
            CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
            CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
            Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Fast, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
        eltwise_pipeline<EltwiseOutputPolicy::Bulk, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    } else {
        auto chain = eltwise_chain(
            CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
            CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
            Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Exact, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
        eltwise_pipeline<EltwiseOutputPolicy::Bulk, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    }
#endif

#ifdef INP_FLOAT
    // BFloat16 path: keep the FPU mul (DestReuse) for perf parity with the
    // original kernel. CopyTile waits on cb_input but does NOT pop; the
    // DestReuseMul re-uses the same waited tile (NoWaitPop policy) and pops
    // it after the multiply. Two physical CB touches collapse into one
    // wait + one pop per iteration.
    using namespace compute_kernel_lib::eltwise;
    if (use_approx) {
        auto chain = eltwise_chain(
            CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
            Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Fast, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            DestReuseMul<cb_input, Dst::D0, DestReuseInputPolicy::NoWaitPop>{});
        eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    } else {
        auto chain = eltwise_chain(
            CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
            Exp<Approx::Exact, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Exact, Dst::D0>{},
            Tanh<Approx::Exact, Dst::D0>{},
            DestReuseMul<cb_input, Dst::D0, DestReuseInputPolicy::NoWaitPop>{});
        eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
    }
#endif
}
