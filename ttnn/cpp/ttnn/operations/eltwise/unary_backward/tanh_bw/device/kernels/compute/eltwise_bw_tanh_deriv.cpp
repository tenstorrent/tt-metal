// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for tanh backward using sech²(x) = 4·exp(-2|x|) / (1 + exp(-2|x|))²
// Avoids catastrophic cancellation in the naive 1 - tanh²(x) formula.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_grad_out = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_1);
    constexpr uint32_t cb_grad_in = static_cast<uint32_t>(tt::CBIndex::c_2);

    init_sfpu(cb_grad_out, cb_grad_in);

    // grad_in = grad_out * sech²(input). Both inputs are block-waited upfront
    // so the pipeline can index into them with WaitUpfrontPopAtEnd Loads.
    auto chain = sfpu_chain(
        Load<cb_grad_out, Dst::D0, LoadPolicy::WaitUpfrontPopAtEnd>{},
        Load<cb_input, Dst::D1, LoadPolicy::WaitUpfrontPopAtEnd>{},
        TanhDerivative<Approx::Exact, Dst::D1>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        sfpu_pipeline<SfpuOutputPolicy::Bulk, SfpuDataFormatReconfig::NONE>(chain, cb_grad_in, per_core_block_size);
    }
}
