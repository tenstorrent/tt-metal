// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr int onetile = 1;
    const auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    DataflowBuffer dfb_scaler(dfb::scaler);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        ckl::mul<
            ckl::input(dfb::in0),
            ckl::input(dfb::in1),
            ckl::output(
                dfb::im0, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
            ckl::IterationShape::tiles(onetile));

        // reduce-w
        if (last_out) {
            ckl::reduce<
                REDUCE_OP,
                REDUCE_DIM,
                dfb::im0,
                dfb::scaler,
                dfb::out,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::NONE>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(dfb::im1, block));
        } else {
            ckl::reduce<
                REDUCE_OP,
                REDUCE_DIM,
                dfb::im0,
                dfb::scaler,
                dfb::im1,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::NONE>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(dfb::im1, block));
        }
    }
    // The reduce helper reuses the scaler tile for every block.
    dfb_scaler.pop_front(onetile);
}
