// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr int onetile = 1;
    const auto has_input_grad = get_arg(args::has_input_grad);
    const auto has_other_grad = get_arg(args::has_other_grad);
    const auto per_core_block_cnt = get_arg(args::per_core_block_cnt);

    DataflowBuffer dfb_c0(dfb::in0);

    compute_kernel_hw_startup(dfb::in2, dfb::in0, dfb::out0);
    dfb_c0.wait_front(onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            ckl::mul<
                ckl::input(
                    dfb::in2, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb::in0,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb::out0,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::tiles(onetile));
        }

        if (has_other_grad) {
            ckl::mul<
                ckl::input(
                    dfb::in1, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::input(
                    dfb::in0,
                    ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    dfb::out1,
                    ckl::ReservePolicy::PerTile,
                    ckl::PushPolicy::PerTile,
                    ckl::DataFormatReconfig::Disabled)>(ckl::IterationShape::tiles(onetile));
        }
    }
}
