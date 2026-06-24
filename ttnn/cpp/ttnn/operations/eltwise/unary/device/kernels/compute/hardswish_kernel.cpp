// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Hardsigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // MulBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

// `#ifdef`-driven constexpr selectors: keep the program-factory's INP_FLOAT*
// numeric defines but reduce them to compile-time booleans here so the chain
// body is a single eltwise_chain call gated by OptionalChainElement instead
// of two top-level #ifdef'd chain invocations.
#ifdef INP_FLOAT32
constexpr bool kIsFloat32 = true;
#else
constexpr bool kIsFloat32 = false;
#endif
constexpr bool kIsFloat = !kIsFloat32;

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    // Hardswish: x * hardsigmoid(x). Same shape as tanhshrink (27bcccc7fea)
    // with Tanh -> Hardsigmoid and Sub -> Mul.
    //
    // FLOAT32: CopyTile(cb_input -> D0 InputLifecycle::HeldStream) + Hardsigmoid<D0>
    //          + CopyTile(cb_input -> D1 InputLifecycle::NoWaitPop) + MulBinary<D0, D1, D0>
    //          + PackTile.
    // FLOAT:   CopyTile(cb_input -> D0 InputLifecycle::HeldStream) + Hardsigmoid<D0>
    //          + DestReuseBinary<cb_input, Mul, DEST_TO_SRCA> + PackTile.
    //          srca = DEST = hardsigmoid(x), srcb = cb_input,
    //          result = hardsigmoid(x) * cb_input.
    //
    // Why the 2nd CopyTile (cb_input -> D1) is FLOAT32-only:
    //   SFPU MulBinary is pure DEST-to-DEST (mul_binary_tile(dst_in0, dst_in1,
    //   dst_out)) — it cannot read from a CB directly. To compute x *
    //   hardsigmoid(x) on the SFPU we need x present in a DEST slot, so we
    //   load cb_input -> D1 a second time. The FLOAT path avoids this load by
    //   using DestReuseBinary, which is an FPU op (binary_dest_reuse_tiles)
    //   that DOES read a CB on the non-DEST side. Both forms are correct;
    //   the SFPU form is preferred under FP32_DEST_ACC because the FPU's
    //   binary_dest_reuse path loses precision on FP32 DEST inputs.
    //
    // Both paths share the CopyTile(D0 InputLifecycle::HeldStream) + Hardsigmoid + PackTile
    // outer shape. OptionalChainElement collapses the inactive branch to a
    // no-op tag (no wait, no pop, no compute emitted), so the chain "selects"
    // between (CopyTile<D1, InputLifecycle::NoWaitPop> + MulBinary) and DestReuseBinary based
    // on kIsFloat32 / kIsFloat.
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<cb_input, ckl::Dst::D0, ckl::InputLifecycle::HeldStream, ckl::CopyTileReconfig::None>{},
        ckl::Hardsigmoid<ckl::Dst::D0>{},
        ckl::OptionalChainElement<
            kIsFloat32,
            ckl::CopyTile<cb_input, ckl::Dst::D1, ckl::InputLifecycle::NoWaitPop, ckl::CopyTileReconfig::None>>{},
        ckl::OptionalChainElement<kIsFloat32, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>{},
        ckl::OptionalChainElement<
            kIsFloat,
            ckl::DestReuseBinary<cb_input, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>>{},
        ckl::PackTile<cb_output, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
