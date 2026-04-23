// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;
    constexpr auto cb_output = tt::CBIndex::c_2;

    using namespace compute_kernel_lib;

    init_sfpu(cb_input, cb_output);

    // logit(x) = log(x / (1 - x))
    // Pass 1 (if CLAMP): clamp x into cb_tmp, otherwise use cb_input directly.
    // Pass 2: fan-out Load×2 → RsubScalar(D0=1-x) → SfpuDiv(D0=x/(1-x)) → Log(D0).
#ifdef CLAMP
    {
        auto chain1 = sfpu_chain(Load<cb_input, Dst::D0>{}, Clamp<Dst::D0>{packed_scalar1, packed_scalar2});
        eltwise_op<cb_tmp0>(chain1, EltwiseTileShape::flat(num_tiles));
    }
    constexpr auto cb_src = cb_tmp0;
#else
    constexpr auto cb_src = cb_input;
#endif

    auto chain2 = sfpu_chain(
        Load<cb_src, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_src, Dst::D1, LoadPolicy::NoWaitPop>{},
        RsubScalar<Dst::D0>{0x3F800000u},      // D0 = 1.0 - x
        SfpuDiv<Dst::D1, Dst::D0, Dst::D0>{},  // D0 = x / (1-x)
        Log<Dst::D0>{});
    eltwise_op<cb_output>(chain2, EltwiseTileShape::flat(num_tiles));
}
