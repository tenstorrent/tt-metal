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

    // Logsigmoid(x) = -log(1 + exp(-x)) = -softplus(-x).
    //   D0 = cb_input   (InputLifecycle::HeldStream — second copy reuses same tile)
    //   D1 = -cb_input -> exp -> exp(-x) (InputLifecycle::NoWaitPop pops cb_input)
    //   Logsigmoid<D0, D1, D0> reads D0=x and D1=exp(-x), writes D0.
    //   pack_tile(D0) -> cb_output.
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<cb_input, ckl::Dst::D0, ckl::InputLifecycle::HeldStream, ckl::CopyTileReconfig::None>{},
        ckl::CopyTile<cb_input, ckl::Dst::D1, ckl::InputLifecycle::NoWaitPop, ckl::CopyTileReconfig::None>{},
        ckl::Negative<ckl::Dst::D1>{},
        ckl::Exp<ckl::Approx::Fast, ckl::Approx::Fast, ckl::Dst::D1>{},
        ckl::Logsigmoid<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::PackTile<cb_output, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
