// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused d(a*b)/d? backward: input_grad = grad * other, other_grad = grad * input.
// Grad is loaded once per tile and multiplied against each operand; no L1 round-trip
// between the two products.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"  // MulBinary

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(0);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_other = tt::CBIndex::c_2;
    constexpr auto cb_input_grad = tt::CBIndex::c_3;
    constexpr auto cb_other_grad = tt::CBIndex::c_4;

    // Boot unpack from the first input and pack for the first output; the second
    // output's format is applied through DataFormatReconfig::Enabled on its PackTile.
    compute_kernel_hw_startup(cb_grad_out, cb_input_grad);

    for (uint32_t tile = 0; tile < per_core_tile_cnt; ++tile) {
        // Chain 1: input_grad = grad * other.
        // grad_out uses PopPolicy::None so chain 2 can consume the same tile.
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
            ckl::CopyTile<
                ckl::input(
                    cb_other,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled),
                ckl::Dst::D1>{},
            ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                cb_input_grad,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Enabled)>{});

        // Chain 2: other_grad = grad * input. grad_out was left in the CB by chain 1;
        // this chain pops it.
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
            ckl::CopyTile<
                ckl::input(
                    cb_input,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    ckl::InputTileMapping::Scalar,
                    ckl::DataFormatReconfig::Enabled),
                ckl::Dst::D1>{},
            ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(
                cb_other_grad,
                ckl::ReservePolicy::PerTile,
                ckl::PushPolicy::PerTile,
                ckl::DataFormatReconfig::Enabled)>{});
    }
}
