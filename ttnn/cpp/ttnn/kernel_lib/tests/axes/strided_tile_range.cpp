// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t input_stride = get_compile_time_arg_val(2);
    constexpr uint32_t output_stride = get_compile_time_arg_val(3);
    constexpr uint32_t input_base = get_compile_time_arg_val(4);
    constexpr uint32_t output_base = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(cb_in, cb_out);

    cb_wait_front(cb_in, Ht * input_stride);
    cb_reserve_back(cb_out, Ht * output_stride);

    using namespace compute_kernel_lib;
    eltwise_chain(
        EltwiseShape::grid(Ht, Wt),
        CopyTile<input(
            cb_in,
            InputLifecycle::CallerManaged,
            OperandKind::Block,
            DataFormatReconfig::Disabled,
            TileOffset::Strided)>{StridedTileRange{input_base, input_stride}},
        PackTile<output(
            cb_out,
            OutputLifecycle::CallerManaged,
            DataFormatReconfig::Disabled,
            PackRelu::Disabled,
            L1Accumulation::Disabled,
            DestAccumulation::Disabled,
            TileOffset::Strided)>{StridedTileRange{output_base, output_stride}});

    cb_pop_front(cb_in, Ht * input_stride);
    cb_push_back(cb_out, Ht * output_stride);
}
