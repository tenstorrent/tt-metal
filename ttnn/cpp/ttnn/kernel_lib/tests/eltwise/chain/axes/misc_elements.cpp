// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Small element scenarios that do not justify separate kernel files:
//   0: ternary Where over three input CBs;
//   1: one ReLU pack plus one unmodified pack.
//
// CT args: [n, mode].

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/special.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);
    static_assert(mode < 2);

    using namespace compute_kernel_lib;
    if constexpr (mode == 0) {
        compute_kernel_hw_startup(cb_a, cb_b, cb_out);
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_a)>{},
            CopyTile<input(cb_b), Dst::D1>{},
            CopyTile<input(cb_c), Dst::D2>{},
            Where<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{},
            PackTile<output(cb_out)>{});
    } else {
        constexpr uint32_t cb_linear = tt::CBIndex::c_17;
        compute_kernel_hw_startup(cb_a, cb_out);
        eltwise_chain(
            IterationShape::tiles(n),
            CopyTile<input(cb_a)>{},
            PackTile<output(
                cb_out,
                ReservePolicy::PerTile,
                PushPolicy::PerTile,
                DataFormatReconfig::Enabled,
                TileAddressing::Direct,
                DestAccumulation::Disabled,
                L1Accumulation::Disabled,
                PackRelu::Zero)>{},
            PackTile<output(cb_linear)>{});
    }
}
