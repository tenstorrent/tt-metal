// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Negative compile test: all PackTile writers share the packer-global accumulation mode.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    using namespace compute_kernel_lib;
    using PreloadedPack = PackTile<output(
        cb_out,
        OutputLifecycle::CallerManaged,
        DataFormatReconfig::Enabled,
        PackRelu::Disabled,
        L1Accumulation::Enabled)>;
    using SeedFirstPack = PackTile<
        output(
            cb_out,
            OutputLifecycle::CallerManaged,
            DataFormatReconfig::Enabled,
            PackRelu::Disabled,
            L1Accumulation::SeedFirst),
        Dst::D1>;
    eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_in)>{}, PreloadedPack{}, SeedFirstPack{});
}
