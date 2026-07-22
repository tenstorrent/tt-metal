// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: reserve-one/push-one must be emitted only once for an accumulator CB.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    using namespace compute_kernel_lib;
    using L1Pack0 = PackTile<output(
        cb_out,
        OutputLifecycle::L1Accumulation,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Enabled)>;
    using L1Pack1 = PackTile<
        output(
            cb_out,
            OutputLifecycle::L1Accumulation,
            DataFormatReconfig::Enabled,
            PackRelu::Disabled,
            L1Accumulation::Enabled),
        Dst::D1>;
    eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_in)>{}, L1Pack0{}, L1Pack1{});
}
