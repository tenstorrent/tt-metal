// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused d(rsub(a,b))/d?  where rsub(a,b) = b - a. So input_grad = -grad, other_grad = grad.
// Mirror of sub_bw's kernel with the negate on the input slot instead of the other slot.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input_grad = tt::CBIndex::c_3;
    constexpr auto cb_other_grad = tt::CBIndex::c_4;

    compute_kernel_hw_startup(cb_grad_out, cb_input_grad);

    for (uint32_t tile = 0; tile < per_core_tile_cnt; ++tile) {
        // Chain 1: input_grad = -grad.
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    cb_grad_out,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled),
                ckl::Dst::D0>{},
            ckl::Negative<ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                cb_input_grad,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Enabled)>{});

        // Chain 2: other_grad = grad (identity).
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::CopyTile<
                ckl::input(
                    cb_grad_out,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::PerTile,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled),
                ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                cb_other_grad,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Enabled)>{});
    }
}
