// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_2;

    constexpr float M_PI = 3.14159265358979323846f;

    using namespace compute_kernel_lib::eltwise;

    init_sfpu(cb_input, cb_output);

    // lgamma_fast(x) = lgamma_stirling(x) + reflection-formula adjustment.
    //
    // DEST layout per iteration:
    //   D0 = lgamma_stirling(x)
    //   D1 = log|sin(pi * frac(x))|, with the integer-x case zeroed
    //   D2 = x   (overwritten twice — first as eq input, then as adjusted's x)
    //   D3 = floor(x), then reused as the 0.0f "else" arm of where
    //
    // cb_input feeds five CopyTiles. The first waits without popping; the
    // intermediate four skip the wait; the final one pops the tile.
    FillScalar<Dst::D2> fill_pi{{}, /*value=*/M_PI};
    FillScalar<Dst::D3> fill_zero{{}, /*value=*/0.0f};

    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitNoPop>{},
        Lgamma<Dst::D0>{},
        fill_pi,
        Frac<Dst::D1>{},
        SfpuMul<Dst::D1, Dst::D2, Dst::D1>{},
        Sin<Dst::D1>{},
        CopyTile<cb_input, Dst::D2, CopyTilePolicy::NoWaitNoPop>{},
        CopyTile<cb_input, Dst::D3, CopyTilePolicy::NoWaitNoPop>{},
        Floor<Dst::D3>{},
        SfpuEq<Dst::D2, Dst::D3, Dst::D2>{},
        fill_zero,
        Where<DataFormat::Float16_b, Dst::D2, Dst::D3, Dst::D1, Dst::D1>{},
        Abs<Dst::D1>{},
        Log<Approx::Exact, Dst::D1>{},
        CopyTile<cb_input, Dst::D2, CopyTilePolicy::NoWaitPop>{},
        LgammaAdjusted<Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});

    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
}
