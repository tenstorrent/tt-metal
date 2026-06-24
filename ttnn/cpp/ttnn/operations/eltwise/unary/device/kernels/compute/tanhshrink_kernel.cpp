// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Tanh
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // SubBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

// #ifdef-driven constexpr selectors: collapse FLOAT / FLOAT32 fork into a
// single chain call gated by OptionalChainElement.
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

    // Tanhshrink: x - tanh(x).
    //
    // Common prefix: D0 = tanh(cb_input).
    //   FLOAT32 tail: D1 = cb_input (InputLifecycle::NoWaitPop) ; SubBinary<D1, D0, D0> -> D0 = x - tanh(x).
    //   FLOAT tail:   DestReuseBinary<cb_input, Sub, DEST_TO_SRCB> with srcb = DEST = tanh(x),
    //                 srca = cb_input from CB ; result = cb_input - tanh(cb_input) -> D0.
    //
    // OptionalChainElement<kIsFloat32 / kIsFloat, ...> collapses the inactive branch to a
    // no-op tag (no wait, no pop, no compute emitted). Single chain replaces the two
    // top-level #ifdef'd chain calls.
    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<cb_input, ckl::Dst::D0, ckl::InputLifecycle::HeldStream, ckl::CopyTileReconfig::None>{},
        ckl::Tanh<ckl::Dst::D0>{},
        ckl::OptionalChainElement<
            kIsFloat32,
            ckl::CopyTile<cb_input, ckl::Dst::D1, ckl::InputLifecycle::NoWaitPop, ckl::CopyTileReconfig::None>>{},
        ckl::OptionalChainElement<kIsFloat32, ckl::SubBinary<ckl::Dst::D1, ckl::Dst::D0, ckl::Dst::D0>>{},
        ckl::OptionalChainElement<
            kIsFloat,
            ckl::DestReuseBinary<cb_input, ckl::BinaryFpuOp::Sub, ckl::DestReuseType::DEST_TO_SRCB>>{},
        ckl::PackTile<cb_output, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::None>{});
}
