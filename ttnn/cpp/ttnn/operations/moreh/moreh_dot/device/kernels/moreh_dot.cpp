// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        ckl::mul<
            ckl::input(tt::CBIndex::c_0),
            ckl::input(tt::CBIndex::c_1),
            ckl::output(tt::CBIndex::c_24, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
            ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));

        // reduce-w
        if (last_out) {
            ckl::reduce<
                REDUCE_OP,
                REDUCE_DIM,
                tt::CBIndex::c_24,
                tt::CBIndex::c_2,
                tt::CBIndex::c_16,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::NONE>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(tt::CBIndex::c_25, block));
        } else {
            ckl::reduce<
                REDUCE_OP,
                REDUCE_DIM,
                tt::CBIndex::c_24,
                tt::CBIndex::c_2,
                tt::CBIndex::c_25,
                ckl::ReduceInputPolicy::WaitAndPopPerTile,
                ckl::ReduceDataFormatReconfigMode::NONE>(
                ckl::ReduceInputBlockShape::single(),
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::Accumulate::at(tt::CBIndex::c_25, block));
        }
    }
}
