// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr int onetile = 1;
    uint32_t has_input_grad = get_arg_val<uint32_t>(0);
    uint32_t has_other_grad = get_arg_val<uint32_t>(1);
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(tt::CBIndex::c_2, tt::CBIndex::c_0, tt::CBIndex::c_16);
    cb_wait_front(tt::CBIndex::c_0, onetile);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        if (has_input_grad) {
            ckl::mul<
                tt::CBIndex::c_2,
                tt::CBIndex::c_0,
                tt::CBIndex::c_16,
                ckl::BroadcastDim::Scalar,
                ckl::input(ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
                ckl::input(ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        }

        if (has_other_grad) {
            ckl::mul<
                tt::CBIndex::c_1,
                tt::CBIndex::c_0,
                tt::CBIndex::c_17,
                ckl::BroadcastDim::Scalar,
                ckl::input(ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
                ckl::input(ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
                ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>(
                ckl::EltwiseShape::tiles(onetile));
        }
    }
}
