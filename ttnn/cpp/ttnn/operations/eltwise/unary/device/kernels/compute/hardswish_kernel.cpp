// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Hardsigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

namespace ckl = compute_kernel_lib;

constexpr bool kIsFloat32 = get_compile_time_arg_val(0) == 1;
constexpr bool kIsFloat = get_compile_time_arg_val(1) == 1;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    compute_kernel_hw_startup(cb_input, cb_output);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(cb_input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::Hardsigmoid<ckl::Dst::D0>{},
        ckl::OptionalChainElement<
            kIsFloat32,
            ckl::CopyTile<
                ckl::input(cb_input, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>>{},
        ckl::OptionalChainElement<kIsFloat32, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>{},
        ckl::OptionalChainElement<
            kIsFloat,
            ckl::DestReuseBinary<
                ckl::input(cb_input, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::BinaryFpuOp::Mul,
                ckl::DestReuseType::DEST_TO_SRCA>>{},
        ckl::PackTile<ckl::output(
            cb_output, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{});
}
