// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"  // Logsigmoid
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

// logsigmoid(x) = min(x, 0) - log1p(exp(-|x|)), computed in place in DST 0 by one SFPU pass.

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in_obj(dfb::in);
    DataflowBuffer dfb_out_obj(dfb::out);
    compute_kernel_hw_startup(dfb::in, dfb::out);

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(dfb::in, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{dfb_in_obj},
        ckl::Logsigmoid<ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb::out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>{
            dfb_out_obj});
}
