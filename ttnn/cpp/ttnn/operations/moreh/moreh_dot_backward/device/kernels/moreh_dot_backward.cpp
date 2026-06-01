// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(tt::CBIndex::c_2, tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_wait_front(tt::CBIndex::c_0, onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            // cb_16 = cb_2 * cb_0 (scalar bcast on cb_0)
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    tt::CBIndex::c_2,
                    tt::CBIndex::c_0,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::Scalar,
                    compute_kernel_lib::BinaryDataFormatReconfig::None,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    tt::CBIndex::c_16,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        }

        if (has_other_grad) {
            // cb_17 = cb_1 * cb_0 (scalar bcast on cb_0)
            compute_kernel_lib::eltwise_chain(
                onetile,
                compute_kernel_lib::BinaryFpu<
                    tt::CBIndex::c_1,
                    tt::CBIndex::c_0,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::Scalar,
                    compute_kernel_lib::BinaryDataFormatReconfig::None,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>{},
                compute_kernel_lib::PackTile<
                    tt::CBIndex::c_17,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OutputLifecycle::Streaming,
                    compute_kernel_lib::PackTileReconfig::None>{});
        }
    }
}
