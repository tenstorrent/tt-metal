// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"    // Log
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"  // Clamp, RsubUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

namespace ckl = compute_kernel_lib;

// Formula: logit(x) = log(x/(1-x)) -- calls clamp, rsub, div, and log tiles.

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;
    constexpr auto cb_output = tt::CBIndex::c_2;

    compute_kernel_hw_startup(cb_input, cb_tmp0);

#ifdef CLAMP
    constexpr bool do_clamp = true;
#else
    constexpr bool do_clamp = false;
#endif
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::OptionalChainElement<do_clamp, ckl::Clamp<ckl::Dst::D0>>{packed_scalar1, packed_scalar2},
        ckl::PackTile<ckl::output(
            cb_tmp0, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(cb_tmp0, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(cb_tmp0, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::RsubUnary<ckl::Dst::D0>{0x3F800000u},  // 1.0 - x
        ckl::DivBinary<ckl::Dst::D1, ckl::Dst::D0, ckl::Dst::D0>{},
        ckl::Log<ckl::Approx::Exact, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            cb_output, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
