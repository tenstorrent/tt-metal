// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"  // Hardsigmoid
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

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

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::CopyTile<
            ckl::input(cb_input, ckl::InputLifecycle::HeldStream, ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::Hardsigmoid<ckl::Dst::D0>{},
        ckl::OptionalChainElement<
            kIsFloat32,
            ckl::CopyTile<
                ckl::input(cb_input, ckl::InputLifecycle::NoWaitPop, ckl::DataFormatReconfig::Disabled),
                ckl::Dst::D1>>{},
        ckl::OptionalChainElement<kIsFloat32, ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>>{},
        ckl::OptionalChainElement<
            kIsFloat,
            ckl::DestReuseBinary<ckl::input(cb_input), ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>>{},
        ckl::PackTile<ckl::output(cb_output, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}
