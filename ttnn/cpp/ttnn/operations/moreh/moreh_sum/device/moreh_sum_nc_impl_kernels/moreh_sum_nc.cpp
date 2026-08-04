// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    namespace ckl = compute_kernel_lib;

    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(num_output_tiles, num_input_tiles),
        ckl::BinaryFpu<
            ckl::input(cb_in0, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::input(cb_in1, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Scalar),
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::Dst::D0,
            ckl::DestAccumulation::PerRow>{},
        ckl::PackTile<ckl::output(
            cb_out0,
            ckl::ReservePolicy::PerOuter,
            ckl::PushPolicy::PerOuter,
            ckl::DataFormatReconfig::Enabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::PerRow)>{});
}
