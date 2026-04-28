// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

// y = sin(x). Smoke test for eltwise_trig.

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;

    using namespace compute_kernel_lib::eltwise;

    init_sfpu(cb_in, cb_out);

    auto chain = eltwise_chain(CopyTile<cb_in, Dst::D0>{}, Sin<Dst::D0>{});
    eltwise_pipeline(chain, cb_out, num_tiles);
}
