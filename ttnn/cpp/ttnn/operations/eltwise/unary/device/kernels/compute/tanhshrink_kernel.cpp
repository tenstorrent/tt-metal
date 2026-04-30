// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_input, cb_output);
    ckernel::init_sfpu(cb_input, cb_output);

    // tanhshrink(x) = x - tanh(x). Same chain works for both INP_FLOAT and
    // INP_FLOAT32 — V2 helper handles same-CB dedup so cb_input is waited
    // and popped exactly once per iteration despite two CopyTile elements.
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;
    using compute_kernel_lib::SubBinary;
    using compute_kernel_lib::Tanh;

    eltwise_pipeline<cb_output>(
        num_tiles,
        eltwise_chain(
            CopyTile<cb_input, Dst::D0>{},
            CopyTile<cb_input, Dst::D1>{},
            Tanh<compute_kernel_lib::Approx::Exact, Dst::D1>{},
            SubBinary<Dst::D0, Dst::D1, Dst::D0>{}));
}
