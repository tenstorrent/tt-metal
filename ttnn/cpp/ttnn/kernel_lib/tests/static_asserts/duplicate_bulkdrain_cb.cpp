// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: BulkDrain waits for the full input window upfront, so
// two independently managed readers cannot share the same CB front.
// MUST fail to compile with "two CB-reader elements share a CB".

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<input(cb_in, WaitPolicy::Upfront, PopPolicy::PerTile), Dst::D0>{},
        CopyTile<input(cb_in, WaitPolicy::Upfront, PopPolicy::PerTile), Dst::D1>{},
        PackTile<output(cb_out)>{});
}
