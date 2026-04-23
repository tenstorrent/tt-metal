// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    init_sfpu(cb_input, cb_output);

    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D1, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D2, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D3, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D4, LoadPolicy::NoWaitPop>{},
        LgammaStirling<Dst::D0>{},
        FillTile<Dst::D5>{3.14159265358979323846f},
        Frac<Dst::D1>{},
        SfpuMul<Dst::D1, Dst::D5, Dst::D1>{},
        Sin<Dst::D1>{},
        Floor<Dst::D3>{},
        SfpuEq<Dst::D2, Dst::D3, Dst::D2>{},
        FillTile<Dst::D3>{0.0f},
        Where<DataFormat::Float16_b, Dst::D2, Dst::D3, Dst::D1, Dst::D1>{},
        Abs<Dst::D1>{},
        Log<Dst::D1>{},
        LgammaAdjusted<Dst::D0, Dst::D1, Dst::D4, Dst::D0>{});

    eltwise_op<cb_output, Dst::D0, EltwiseOutputPolicy::Bulk>(chain, EltwiseTileShape::flat(num_tiles));
}
