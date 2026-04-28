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

    // FP32 lgamma — uses the lgamma_stirling_float_tile kernel which expects
    // x and log(z = (x<0.5 ? 1-x : x)) and writes lgamma_stirling(x) into Out,
    // followed by the same reflection-formula adjustment as lgamma_fast.
    //
    // DEST footprint is 4 slots (D0..D3) — exactly the FP32 + half-sync cap.
    FillScalar<Dst::D2> fill_half{{}, /*value=*/0.5f};
    FillScalar<Dst::D2> fill_one_d2{{}, /*value=*/1.0f};
    FillScalar<Dst::D2> fill_pi_d2{{}, /*value=*/M_PI};
    FillScalar<Dst::D3> fill_zero_d3{{}, /*value=*/0.0f};

    auto chain = eltwise_chain(
        CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitNoPop>{},
        fill_half,
        SfpuSub<Dst::D1, Dst::D2, Dst::D1>{},
        Ltz<Dst::D1>{},
        fill_one_d2,
        SfpuSub<Dst::D2, Dst::D0, Dst::D2>{},
        Where<DataFormat::Float32, Dst::D1, Dst::D2, Dst::D0, Dst::D1>{},
        Log<Approx::Exact, Dst::D1>{},
        LgammaStirlingFloat<Dst::D0, Dst::D1, Dst::D0>{},
        fill_pi_d2,
        CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitNoPop>{},
        Frac<Dst::D1>{},
        SfpuMul<Dst::D1, Dst::D2, Dst::D1>{},
        Sin<Dst::D1>{},
        CopyTile<cb_input, Dst::D2, CopyTilePolicy::NoWaitNoPop>{},
        CopyTile<cb_input, Dst::D3, CopyTilePolicy::NoWaitNoPop>{},
        Floor<Dst::D3>{},
        SfpuEq<Dst::D2, Dst::D3, Dst::D2>{},
        fill_zero_d3,
        Where<DataFormat::Float32, Dst::D2, Dst::D3, Dst::D1, Dst::D1>{},
        Abs<Dst::D1>{},
        Log<Approx::Exact, Dst::D1>{},
        CopyTile<cb_input, Dst::D2, CopyTilePolicy::NoWaitPop>{},
        LgammaAdjusted<Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});

    eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
}
