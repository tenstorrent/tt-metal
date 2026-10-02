// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_x = 0;
    constexpr uint32_t cb_clip_coef_clamped = 1;  // clip_coef_clamped
    constexpr uint32_t cb_y = 16;

    compute_kernel_hw_startup(cb_x, cb_clip_coef_clamped, cb_y);

    ckl::mul<
        ckl::input(cb_x, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
        ckl::input(
            cb_clip_coef_clamped,
            ckl::BroadcastDim::Scalar,
            ckl::WaitPolicy::Upfront,
            ckl::PopPolicy::AtEnd,
            ckl::DataFormatReconfig::Disabled),
        ckl::output(cb_y, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
        ckl::IterationShape::tiles(num_tiles));
}
