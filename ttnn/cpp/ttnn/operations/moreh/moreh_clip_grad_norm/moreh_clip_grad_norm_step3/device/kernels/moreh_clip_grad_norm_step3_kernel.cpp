// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t dfb_x_id = 0;
    constexpr uint32_t dfb_clip_coef_clamped_id = 1;  // clip_coef_clamped
    constexpr uint32_t dfb_y_id = 16;

    compute_kernel_hw_startup(dfb_x_id, dfb_clip_coef_clamped_id, dfb_y_id);

    ckl::mul<
        ckl::input(dfb_x_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::input(
            dfb_clip_coef_clamped_id,
            ckl::BroadcastDim::Scalar,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::DataFormatReconfig::Disabled),
        ckl::output(
            dfb_y_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(num_tiles));
}
