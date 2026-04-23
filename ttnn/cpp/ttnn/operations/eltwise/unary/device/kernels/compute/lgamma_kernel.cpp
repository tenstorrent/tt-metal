// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    const uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    init_sfpu(cb_input, cb_output);

    // lgamma(x) via Stirling + reflection formula.
    // Pre-load x to D0-D5 (fan-out) to avoid CB re-reads within the tile cycle.
    // D6 is used as scratch for fill values (0.5, 1.0, π, 0.0).
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D1, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D2, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D3, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D4, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D5, LoadPolicy::NoWaitPop>{},
        FillTile<Dst::D6>{0.5f},
        SfpuSub<Dst::D1, Dst::D6, Dst::D1>{},
        Ltz<Dst::D1>{},
        FillTile<Dst::D6>{1.0f},
        SfpuSub<Dst::D6, Dst::D0, Dst::D6>{},
        Where<DataFormat::Float32, Dst::D1, Dst::D6, Dst::D0, Dst::D1>{},
        Log<Approx::Exact, Dst::D1>{},
        LgammaStirlingFloat<Dst::D0, Dst::D1, Dst::D0>{},
        FillTile<Dst::D6>{3.14159265358979323846f},
        Frac<Dst::D2>{},
        SfpuMul<Dst::D2, Dst::D6, Dst::D2>{},
        Sin<Dst::D2>{},
        Floor<Dst::D3>{},
        SfpuEq<Dst::D4, Dst::D3, Dst::D4>{},
        FillTile<Dst::D3>{0.0f},
        Where<DataFormat::Float32, Dst::D4, Dst::D3, Dst::D2, Dst::D2>{},
        Abs<Dst::D2>{},
        Log<Dst::D2>{},
        LgammaAdjusted<Dst::D0, Dst::D2, Dst::D5, Dst::D0>{});

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(chain, EltwiseTileShape::flat(per_core_block_dim));
    }
}
