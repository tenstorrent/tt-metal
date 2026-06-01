// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        bool last_out = block == (per_core_block_cnt - 1);

        // elemwise-mul: cb_intermed0 = cb_in0 * cb_in1
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                tt::CBIndex::c_0,
                tt::CBIndex::c_1,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                tt::CBIndex::c_24,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::PackTileReconfig::None>{});

        // reduce-w
        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
            tt::CBIndex::c_24,
            tt::CBIndex::c_2,
            last_out ? tt::CBIndex::c_16 : tt::CBIndex::c_25,
            compute_kernel_lib::ReduceInputBlockShape::single(),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::Accumulate::at(tt::CBIndex::c_25, block));
    }
}
