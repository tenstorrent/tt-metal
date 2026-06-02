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
    using namespace compute_kernel_lib;

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    // Mish: x * tanh(softplus(x)) = x * tanh(log1p(exp(x))).
    //
    // Shared chain prefix / suffix (held vs streamed lifecycle, tanh of the
    // softplus result, mul-by-x tail) are identical across the use_approx
    // fork. Only Exp + Log1p differ in their Approx template arg. Define the
    // common elements once as type aliases + value instances so the two chain
    // bodies stay readable and the type system enforces they're identical
    // across branches.
    constexpr CopyTile<cb_input, Dst::D0, InputLifecycle::HeldStream, OperandKind::Scalar, CopyTileReconfig::None>
        load_x_held{};
    constexpr Tanh<Dst::D0> tanh_d0{};
    // FLOAT32 tail: load x again to D1 then SFPU MulBinary.
    constexpr OptionalChainElement<
        kIsFloat32,
        CopyTile<cb_input, Dst::D1, InputLifecycle::NoWaitPop, OperandKind::Scalar, CopyTileReconfig::None>>
        load_x_d1_for_sfpu{};
    constexpr OptionalChainElement<kIsFloat32, MulBinary<Dst::D0, Dst::D1, Dst::D0>> sfpu_mul_d0_d1{};
    // FLOAT tail: DestReuseBinary reads cb_input on srcb, DEST on srca.
    constexpr OptionalChainElement<
        kIsFloat,
        DestReuseBinary<
            cb_input,
            BinaryFpuOp::Mul,
            DestReuseType::DEST_TO_SRCA,
            Dst::D0,
            Dst::D0,
            DestReuseReconfig::Input,
            InputLifecycle::Streaming,
            OperandKind::Scalar>>
        fpu_mul_dest_x{};
    constexpr PackTile<cb_output, OutputLifecycle::Streaming, PackTileReconfig::None> pack_y{};

    if (use_approx) {
        eltwise_chain(
            num_tiles,
            load_x_held,
            Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
            Log1p<Approx::Fast, Dst::D0>{},
            tanh_d0,
            load_x_d1_for_sfpu,
            sfpu_mul_d0_d1,
            fpu_mul_dest_x,
            pack_y);
    } else {
        eltwise_chain(
            num_tiles,
            load_x_held,
            Exp<Approx::Exact, Approx::Exact, Dst::D0>{},
            Log1p<Approx::Exact, Dst::D0>{},
            tanh_d0,
            load_x_d1_for_sfpu,
            sfpu_mul_d0_d1,
            fpu_mul_dest_x,
            pack_y);
    }
}
