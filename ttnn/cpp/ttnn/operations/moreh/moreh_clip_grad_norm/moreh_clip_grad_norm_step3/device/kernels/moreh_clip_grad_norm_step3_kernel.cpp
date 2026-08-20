// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t dfb_x_id = 0;
    constexpr uint32_t dfb_clip_coef_clamped_id = 1;  // clip_coef_clamped
    constexpr uint32_t dfb_y_id = 16;

    compute_kernel_hw_startup(dfb_x_id, dfb_clip_coef_clamped_id, dfb_y_id);

    compute_kernel_lib::mul<
        compute_kernel_lib::input(
            dfb_x_id,
            compute_kernel_lib::WaitPolicy::PerTile,
            compute_kernel_lib::PopPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::input(
            dfb_clip_coef_clamped_id,
            compute_kernel_lib::BroadcastDim::Scalar,
            compute_kernel_lib::WaitPolicy::Upfront,
            compute_kernel_lib::PopPolicy::AtEnd,
            compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            dfb_y_id,
            compute_kernel_lib::ReservePolicy::PerTile,
            compute_kernel_lib::PushPolicy::PerTile,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(compute_kernel_lib::IterationShape::tiles(num_tiles));
}
