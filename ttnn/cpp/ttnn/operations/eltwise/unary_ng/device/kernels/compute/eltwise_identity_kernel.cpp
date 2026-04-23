// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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

    auto chain = sfpu_chain(Load<cb_input, Dst::D0>{});
    eltwise_op<cb_output>(chain, EltwiseTileShape::flat(num_tiles));
}
