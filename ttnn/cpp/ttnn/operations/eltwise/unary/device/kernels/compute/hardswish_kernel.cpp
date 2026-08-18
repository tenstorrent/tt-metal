// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Hardsigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional

namespace ckl = compute_kernel_lib;

constexpr bool kIsFloat32 = get_compile_time_arg_val(0) == 1;
constexpr bool kIsInt = get_compile_time_arg_val(1) == 1;
constexpr bool kIsFloat = !kIsFloat32 && !kIsInt;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(
                dfb_input_id,
                ckl::WaitPolicy::PerTile,
                kIsInt ? ckl::PopPolicy::PerTile : ckl::PopPolicy::None,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::Hardsigmoid<ckl::Dst::D0>{},
        ckl::Optional<
            kIsFloat32,
            ckl::CopyTile<
                ckl::input(
                    dfb_input_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>>{},
        ckl::Optional<kIsFloat32, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>{},
        ckl::Optional<
            kIsFloat,
            ckl::DestReuseBinary<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb_input_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::DestReuseType::DEST_TO_SRCA>>{},
        ckl::PackTile<ckl::output(
            dfb_output_id,
            ckl::ReservePolicy::PerTile,
            ckl::PushPolicy::PerTile,
            ckl::DataFormatReconfig::Disabled)>{});
}
