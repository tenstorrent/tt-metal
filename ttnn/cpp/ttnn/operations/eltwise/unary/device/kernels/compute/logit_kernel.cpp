// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_output = tt::CBIndex::c_2;
    constexpr uint32_t cb_tmp0 = tt::CBIndex::c_1;

    using namespace compute_kernel_lib::eltwise;

    init_sfpu(cb_input, cb_output);

    // Two-stage pipeline interleaved per tile (keeps cb_tmp0 double-buffer
    // discipline matching the original raw-LLK kernel).
    //
    // Stage 1: cb_input -> cb_tmp0           [optional clamp]
    // Stage 2: cb_tmp0  -> cb_output         logit(x) = log(x / (1 - x))
    for (uint32_t i = 0; i < num_tiles; ++i) {
#ifdef CLAMP
        // Aggregate-init through the CRTP base requires nested braces.
        Clamp<Dst::D0> clamp{{}, /*param_min=*/packed_scalar1, /*param_max=*/packed_scalar2};
        auto chain1 = eltwise_chain(CopyTile<cb_input, Dst::D0>{}, clamp);
#else
        auto chain1 = eltwise_chain(CopyTile<cb_input, Dst::D0>{});
#endif
        eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain1, cb_tmp0, 1);

        // Stage 2 — D0 = x, D1 = x; rsub D0 = 1 - x; div D1 / D0; log result.
        Rsub<Dst::D0> rsub_one_minus{{}, /*param0=*/0x3F800000u};
        auto chain2 = eltwise_chain(
            CopyTile<cb_tmp0, Dst::D0, CopyTilePolicy::WaitNoPop>{},
            CopyTile<cb_tmp0, Dst::D1, CopyTilePolicy::NoWaitPop>{},
            rsub_one_minus,
            SfpuDiv<Dst::D1, Dst::D0, Dst::D0>{},
            Log<Approx::Exact, Dst::D0>{});
        eltwise_pipeline<EltwiseOutputPolicy::PerTile, EltwiseDataFormatReconfig::NONE>(chain2, cb_output, 1);
    }
}
