// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Negative
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Logsigmoid
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<
            cb_input,
            ckl::Dst::D0,
            ckl::input(ckl::InputLifecycle::HeldStream, ckl::DataFormatReconfig::Disabled)>{},
        ckl::CopyTile<
            cb_input,
            ckl::Dst::D1,
            ckl::input(ckl::InputLifecycle::NoWaitPop, ckl::DataFormatReconfig::Disabled)>{},
        ckl::Negative<ckl::Dst::D1>{},
        ckl::Exp<ckl::Approx::Fast, ckl::Approx::Fast, ckl::Dst::D1>{},
        ckl::Logsigmoid<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<cb_output, ckl::output(ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}
