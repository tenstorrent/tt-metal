// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"

void kernel_main() {
    namespace ckl = compute_kernel_lib;

    // compile-time args
    // num_output_tiles carries the per-core work-split count (the host's num_cols_per_core_group_N).
    constexpr uint32_t num_output_tiles = get_arg(args::num_output_tiles);
    constexpr uint32_t num_input_tiles = get_arg(args::num_input_tiles);

    constexpr auto dfb_in0_id = dfb::input;
    constexpr auto dfb_in1_id = dfb::zero;
    constexpr auto dfb_out0_id = dfb::out;

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
