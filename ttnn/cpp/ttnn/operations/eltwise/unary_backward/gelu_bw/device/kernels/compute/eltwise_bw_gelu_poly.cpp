// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for GELU backward using polynomial-based GELU derivative
// Uses Sollya-derived minimax polynomials for high accuracy (Max ULP = 1)

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary.hpp"

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_grad_out = tt::CBIndex::c_0;
    constexpr auto cb_input = tt::CBIndex::c_1;
    constexpr auto cb_grad_in = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_grad_out, cb_grad_in);
    ckernel::init_sfpu(cb_grad_out, cb_grad_in);

    using namespace compute_kernel_lib;
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        // Per-block upfront wait/pop on both inputs; chain shape mirrors
        // tanh_bw with GeluDerivative on D1.
        compute_kernel_lib::eltwise_pipeline<cb_grad_in>(
            per_core_block_size,
            compute_kernel_lib::eltwise_chain(
                compute_kernel_lib::
                    CopyTile<cb_grad_out, Dst::D0, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter>{},
                compute_kernel_lib::
                    CopyTile<cb_input, Dst::D1, CopyTilePolicy::WaitUpfrontPopAtEnd, CbIndexMode::BlockIter>{},
                compute_kernel_lib::GeluDerivative<Dst::D1>{},
                compute_kernel_lib::MulBinary<Dst::D0, Dst::D1, Dst::D0>{}));
    }
}
