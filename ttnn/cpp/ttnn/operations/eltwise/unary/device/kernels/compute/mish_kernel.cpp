// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"         // Exp, Log1p
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// #ifdef-driven constexpr selectors: collapse the FLOAT / FLOAT32 fork into a
// single chain call gated by OptionalChainElement. use_approx remains a runtime
// branch (selecting Approx::Fast vs Approx::Exact compile-time templates on
// Exp/Log1p), so the kernel ends up with TWO eltwise_chain instantiations
// (was FOUR before this consolidation).
#ifdef INP_FLOAT32
constexpr bool kIsFloat32 = true;
#else
constexpr bool kIsFloat32 = false;
#endif
constexpr bool kIsFloat = !kIsFloat32;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    // Mish: x * tanh(softplus(x)) = x * tanh(log1p(exp(x))).
    //
    // Common prefix: D0 = tanh(log1p(exp(cb_input))).
    //   FLOAT32 tail: D1 = cb_input (NoWaitPop) + MulBinary<D0, D1, D0>.
    //   FLOAT   tail: DestReuseBinary<cb_input, Mul, DEST_TO_SRCA>
    //                 (srca = cb_input from CB, srcb = DEST = tanh(softplus(x))).
    //
    // OptionalChainElement<kIsFloat32 / kIsFloat, ...> collapses the inactive
    // branch to a no-op tag — no wait, no pop, no compute emitted.
    if (use_approx) {
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
            compute_kernel_lib::OptionalChainElement<
                kIsFloat32,
                compute_kernel_lib::CopyTile<
                    cb_input,
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::NoWaitPop,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::None>>{},
            compute_kernel_lib::OptionalChainElement<
                kIsFloat32,
                compute_kernel_lib::
                    MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>>{},
            compute_kernel_lib::OptionalChainElement<
                kIsFloat,
                compute_kernel_lib::DestReuseBinary<
                    cb_input,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::DestReuseReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::OperandKind::Scalar>>{},
            compute_kernel_lib::PackTile<
                cb_output,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
    } else {
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
            compute_kernel_lib::OptionalChainElement<
                kIsFloat32,
                compute_kernel_lib::CopyTile<
                    cb_input,
                    compute_kernel_lib::Dst::D1,
                    compute_kernel_lib::NoWaitPop,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::None>>{},
            compute_kernel_lib::OptionalChainElement<
                kIsFloat32,
                compute_kernel_lib::
                    MulBinary<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>>{},
            compute_kernel_lib::OptionalChainElement<
                kIsFloat,
                compute_kernel_lib::DestReuseBinary<
                    cb_input,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::DestReuseType::DEST_TO_SRCA,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::DestReuseReconfig::Input,
                    compute_kernel_lib::Streaming,
                    compute_kernel_lib::OperandKind::Scalar>>{},
            compute_kernel_lib::PackTile<
                cb_output,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OutStreaming,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::PackTileReconfig::None>{});
    }
}
