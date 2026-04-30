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
    const uint32_t approx_arg = get_arg_val<uint32_t>(1);
    const bool use_approx = (approx_arg != 0u);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);

    ckernel::compute_kernel_hw_startup(cb_input, cb_output);
    ckernel::init_sfpu(cb_input, cb_output);

    // mish(x) = x * tanh(log1p(exp(x))).
    // Chain: D0=x, D1=x; D1=exp; D1=log1p; D1=tanh; D0=mul(D0,D1).
    // V2 helper handles same-CB dedup. Two paths (use_approx) selected at runtime.
    using compute_kernel_lib::Approx;
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;
    using compute_kernel_lib::Exp;
    using compute_kernel_lib::FP32DestAcc;
    using compute_kernel_lib::Log1p;
    using compute_kernel_lib::MulBinary;
    using compute_kernel_lib::Tanh;

    if (use_approx) {
        eltwise_pipeline<cb_output>(
            num_tiles,
            eltwise_chain(
                CopyTile<cb_input, Dst::D0>{},
                CopyTile<cb_input, Dst::D1>{},
                Exp<Approx::Fast, Approx::Fast, FP32DestAcc::Off, Dst::D1>{},
                Log1p<Dst::D1>{},
                Tanh<Approx::Exact, Dst::D1>{},
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{}));
    } else {
        eltwise_pipeline<cb_output>(
            num_tiles,
            eltwise_chain(
                CopyTile<cb_input, Dst::D0>{},
                CopyTile<cb_input, Dst::D1>{},
                Exp<Approx::Exact, Approx::Fast, FP32DestAcc::Off, Dst::D1>{},
                Log1p<Dst::D1>{},
                Tanh<Approx::Exact, Dst::D1>{},
                MulBinary<Dst::D0, Dst::D1, Dst::D0>{}));
    }
}
