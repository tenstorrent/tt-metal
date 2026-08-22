// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"         // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"         // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Logsigmoid

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto dfb_input_id = tt::CBIndex::c_0;
    constexpr auto dfb_output_id = tt::CBIndex::c_2;

    compute_kernel_hw_startup(dfb_input_id, dfb_output_id);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(dfb_input_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(dfb_input_id, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::Negative<ckl::Dst::D1>{},
        ckl::Exp<ckl::Approx::Fast, ckl::Dst::D1>{},
        ckl::Logsigmoid<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_output_id,
            ckl::ReservePolicy::PerTile,
            ckl::PushPolicy::PerTile,
            ckl::DataFormatReconfig::Disabled)>{});
}
