// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_clip_coef_clamped = 1;
    constexpr uint32_t cb_y = 16;

    compute_kernel_hw_startup(cb_x, cb_clip_coef_clamped, cb_y);

    compute_kernel_lib::mul<
        cb_x,
        cb_clip_coef_clamped,
        cb_y,
        compute_kernel_lib::BroadcastDim::Scalar,
        compute_kernel_lib::input(
            compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::input(
            compute_kernel_lib::InputLifecycle::Bulk, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            compute_kernel_lib::OutputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled)>(
        compute_kernel_lib::EltwiseShape::tiles(num_tiles));
}
