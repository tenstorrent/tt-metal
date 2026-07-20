// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_relu = tt::CBIndex::c_16;
    constexpr uint32_t cb_linear = tt::CBIndex::c_17;
    constexpr uint32_t n = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_in, cb_relu);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::tiles(n),
        CopyTile<cb_in>{},
        PackTile<cb_relu, output(OutputLifecycle::Streaming, DataFormatReconfig::Enabled, PackRelu::Zero)>{},
        PackTile<cb_linear>{});
}
