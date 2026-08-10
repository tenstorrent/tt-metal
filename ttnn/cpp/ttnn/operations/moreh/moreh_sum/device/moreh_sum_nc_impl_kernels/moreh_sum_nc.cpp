// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"

void kernel_main() {
    namespace ckl = compute_kernel_lib;

    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);

    constexpr auto dfb_in0_id = tt::CBIndex::c_0;
    constexpr auto dfb_in1_id = tt::CBIndex::c_1;
    constexpr auto dfb_out0_id = tt::CBIndex::c_16;

    compute_kernel_hw_startup(dfb_in0_id, dfb_in1_id, dfb_out0_id);

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(num_output_tiles, num_input_tiles),
        ckl::BinaryFpu<
            ckl::input(
                dfb_in0_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::input(dfb_in1_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Scalar),
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None,
            ckl::Dst::D0,
            ckl::DestAccumulation::PerRow>{},
        ckl::PackTile<ckl::output(
            dfb_out0_id,
            ckl::ReservePolicy::PerOuter,
            ckl::PushPolicy::PerOuter,
            ckl::DataFormatReconfig::Enabled,
            ckl::PackRelu::Disabled,
            ckl::L1Accumulation::Disabled,
            ckl::DestAccumulation::PerRow)>{});
}
