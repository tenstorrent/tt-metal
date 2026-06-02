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
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::HeldStream,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D1,
            compute_kernel_lib::InputLifecycle::NoWaitPop,
            compute_kernel_lib::OperandKind::Scalar,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::Negative<compute_kernel_lib::Dst::D1>{},
        compute_kernel_lib::
            Exp<compute_kernel_lib::Approx::Fast, compute_kernel_lib::Approx::Fast, compute_kernel_lib::Dst::D1>{},
        compute_kernel_lib::
            Logsigmoid<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
