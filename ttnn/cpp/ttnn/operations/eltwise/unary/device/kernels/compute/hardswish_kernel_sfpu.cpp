// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr uint32_t cb_input = static_cast<uint32_t>(tt::CBIndex::c_0);
    constexpr uint32_t cb_output = static_cast<uint32_t>(tt::CBIndex::c_2);
    init_sfpu(cb_input, cb_output);

    // hardswish(x) = x * hardsigmoid(x)
    auto chain = sfpu_chain(
        Load<cb_input, Dst::D0>{},
        Load<cb_input, Dst::D1>{},
        Hardsigmoid<Dst::D0>{},
        SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        sfpu_pipeline<SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::Bulk>(
            cb_output, /*pack_slot=*/0, per_core_block_dim, chain);
    }
}
