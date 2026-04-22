// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);

    init_sfpu(cb_input, cb_output);

    // logsigmoid(x) = -log(1 + exp(-x))
    // Output CB capacity = 2 → PerTile + Disabled batching.
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},
        Load<cb_input, Dst::D1, LoadPolicy::NoWaitPop>{},
        Neg<Dst::D1>{},
        Exp<Approx::Fast, Approx::Fast, Dst::D1>{},
        Logsigmoid<Dst::D0, Dst::D1, Dst::D0>{});

    sfpu_pipeline<SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::NONE>(chain, cb_output, num_tiles);
}
