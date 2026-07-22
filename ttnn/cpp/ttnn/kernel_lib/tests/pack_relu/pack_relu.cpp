// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

void kernel_main() {
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_sink = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_out2 = tt::CBIndex::c_17;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t mode = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in0, cb_out);

    using namespace compute_kernel_lib;
    using ReluPack = PackTile<output(cb_out, OutputLifecycle::Streaming, DataFormatReconfig::Enabled, PackRelu::Zero)>;

    if constexpr (mode == 0) {
        eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_in0)>{}, ReluPack{});
    } else if constexpr (mode == 1) {
        // A following independent linear pack proves that the first chain restores STACC_RELU.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<input(cb_in0)>{},
            PackTile<output(cb_sink, OutputLifecycle::Streaming, DataFormatReconfig::Enabled, PackRelu::Zero)>{});
        eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_in1)>{}, PackTile<output(cb_out)>{});
    } else if constexpr (mode == 2) {
        eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_in0)>{}, Exp<>{}, ReluPack{});
    } else if constexpr (mode == 3) {
        eltwise_chain(EltwiseShape::tiles(n), CopyTile<input(cb_in0)>{}, Exp<>{}, PackTile<output(cb_out)>{});
    } else {
        // Exercise heterogeneous pack modes in one chain.
        eltwise_chain(
            EltwiseShape::tiles(n),
            CopyTile<input(cb_in0), Dst::D0>{},
            CopyTile<input(cb_in1), Dst::D1>{},
            ReluPack{},
            PackTile<output(cb_out2), Dst::D1>{});
    }
}
