// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_input, cb_output);
    ckernel::init_sfpu(cb_input, cb_output);

    // Fan-out (lessons §3.5): one CB tile drives both D0 (kept) and D1
    // (Negative → Exp → fed as second operand to LogSigmoid). The helper
    // dedups same-CB wait/pop (lessons §3.6) so the cb_input is waited and
    // popped exactly once per iteration regardless of the two CopyTile
    // elements.
    compute_kernel_lib::eltwise_pipeline<cb_output>(
        num_tiles,
        compute_kernel_lib::eltwise_chain(
            compute_kernel_lib::CopyTile<cb_input, compute_kernel_lib::Dst::D0>{},
            compute_kernel_lib::CopyTile<cb_input, compute_kernel_lib::Dst::D1>{},
            compute_kernel_lib::Negative<compute_kernel_lib::Approx::Exact, compute_kernel_lib::Dst::D1>{},
            compute_kernel_lib::Exp<
                compute_kernel_lib::Approx::Fast,
                compute_kernel_lib::Approx::Fast,
                compute_kernel_lib::FP32DestAcc::Off,
                compute_kernel_lib::Dst::D1>{},
            compute_kernel_lib::
                LogSigmoid<compute_kernel_lib::Dst::D0, compute_kernel_lib::Dst::D1, compute_kernel_lib::Dst::D0>{}));
}
