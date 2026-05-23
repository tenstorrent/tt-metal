// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp, Log1p
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    binary_op_init_common(cb_input, cb_input, cb_output);

    // Mish: x * tanh(softplus(x)) = x * tanh(log1p(exp(x))).
    //
    // FLOAT32 path: two CopyTile (D0 HeldStream + D1 NoWaitPop) + Exp + Log1p +
    //   Tanh + MulBinary<D0, D1, D0>.
    //   (Original loaded D0 first, ran SFPU chain on D0, then loaded D1 for mul.
    //   Chain order is identical effect-wise — D1 load doesn't touch D0.)
    //
    // FLOAT path: CopyTile (D0 HeldStream) + Exp + Log1p + Tanh +
    //   DestReuseBinary<cb_input, Mul, DEST_TO_SRCA>.
    //   srca = DEST = tanh(softplus(x)), srcb = cb_input,
    //   result = tanh(softplus(x)) * x.
    //
    // use_approx is a runtime arg, so 4-way dispatch on (approx, dtype). The
    // Exp/Log1p templates use Approx::Fast | Approx::Exact compile-time enum;
    // Tanh in chain is non-templated (uses non-approx LLK like the original).
    if (use_approx) {
#ifdef INP_FLOAT32
        compute_kernel_lib::eltwise_chain(
            num_tiles,
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::
                Exp<compute_kernel_lib::Approx::Fast, compute_kernel_lib::Approx::Fast, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Log1p<compute_kernel_lib::Approx::Fast, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D1,
                compute_kernel_lib::NoWaitPop,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::
                MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_output,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
#endif
#ifdef INP_FLOAT
        compute_kernel_lib::eltwise_chain(
            num_tiles,
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::
                Exp<compute_kernel_lib::Approx::Fast, compute_kernel_lib::Approx::Fast, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Log1p<compute_kernel_lib::Approx::Fast, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::DestReuseBinary<
                cb_input,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::DestReuseReconfig::Input,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_output,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
#endif
    } else {
#ifdef INP_FLOAT32
        compute_kernel_lib::eltwise_chain(
            num_tiles,
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Log1p<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D1,
                compute_kernel_lib::NoWaitPop,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::
                MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::PackTile<
                cb_output,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
#endif
#ifdef INP_FLOAT
        compute_kernel_lib::eltwise_chain(
            num_tiles,
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::HeldStream,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::CopyTileReconfig::None>{},
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Approx::Exact,
                compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Log1p<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::DestReuseBinary<
                cb_input,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::DestReuseReconfig::Input,
                compute_kernel_lib::Streaming,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_output,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
#endif
    }
}
