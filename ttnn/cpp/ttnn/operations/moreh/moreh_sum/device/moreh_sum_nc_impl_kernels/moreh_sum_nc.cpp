// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"

void kernel_main() {
    namespace ckl = compute_kernel_lib;

    // compile-time args
    // num_output_tiles carries the per-core work-split count (the host's num_cols_per_core_group_N).
    constexpr uint32_t num_output_tiles = get_arg(args::num_output_tiles);
    constexpr uint32_t num_input_tiles = get_arg(args::num_input_tiles);

    compute_kernel_hw_startup(dfb::input, dfb::zero, dfb::out);

    ckl::eltwise_chain(
        ckl::IterationShape::grid(num_output_tiles, num_input_tiles),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Add,
            ckl::input(
                dfb::input, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(dfb::zero, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Scalar),
            ckl::Dst::D0,
            ckl::DestAccumulation::PerRow>{},
        ckl::PackTile<ckl::output(
            dfb::out,
            ckl::ReservePolicy::PerOuter,
            ckl::PushPolicy::PerOuter,
            ckl::DataFormatReconfig::Enabled,
            ckl::TileAddressing::Direct,
            ckl::DestAccumulation::PerRow)>{});
}
