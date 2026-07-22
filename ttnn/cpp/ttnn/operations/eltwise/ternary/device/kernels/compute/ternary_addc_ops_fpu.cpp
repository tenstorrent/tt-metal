// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"  // MulUnary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

namespace ckl = compute_kernel_lib;

inline void run_addcmul(uint32_t num_tiles, uint32_t scalar_arg) {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckl::eltwise_chain(
        ckl::EltwiseShape::tiles(num_tiles),
        ckl::BinaryFpu<
            ckl::input(cb_in1, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
            ckl::input(cb_in2, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::None>{},
        ckl::runtime_if(scalar_arg != 1u, ckl::MulUnary<ckl::Dst::D0>{scalar_arg}),
        ckl::DestReuseBinary<ckl::input(cb_in0), ckl::BinaryFpuOp::Add, ckl::DestReuseType::DEST_TO_SRCA>{},
        ckl::PackTile<ckl::output(cb_out, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    binary_op_init_common(cb_in1, cb_in2, cb_out);

    run_addcmul(num_tiles, scalar_arg);
}
