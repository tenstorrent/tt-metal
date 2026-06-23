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
    compute_kernel_lib::eltwise_chain(
        num_tiles,
        compute_kernel_lib::CopyTile<
            cb_input,
            compute_kernel_lib::Dst::D0,
            compute_kernel_lib::InputLifecycle::HeldStream,
            compute_kernel_lib::CopyTileReconfig::None>{},
        compute_kernel_lib::Tanh<compute_kernel_lib::Dst::D0>{},
        compute_kernel_lib::OptionalChainElement<
            kIsFloat32,
            compute_kernel_lib::CopyTile<
                cb_input,
                compute_kernel_lib::Dst::D1,
                compute_kernel_lib::InputLifecycle::NoWaitPop,
                compute_kernel_lib::CopyTileReconfig::None>>{},
        compute_kernel_lib::OptionalChainElement<
            kIsFloat32,
            compute_kernel_lib::
                SubBinary<compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D0>>{},
        compute_kernel_lib::OptionalChainElement<
            kIsFloat,
            compute_kernel_lib::DestReuseBinary<
                cb_input,
                compute_kernel_lib::BinaryFpuOp::Sub,
                compute_kernel_lib::DestReuseType::DEST_TO_SRCB>>{},
        compute_kernel_lib::PackTile<
            cb_output,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::PackTileReconfig::None>{});
}
