// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 2-arg combined reconfig_data_format(curr_a, curr_b) (no _with_dt; unconditional reprogram).
//
// BinaryFpu(CbA,CbB) -> PackTile(CbOut). Element 0 is first, so both srca/srcb are first-emit
// (NO_PREV_DFB) and the chain emits the 2-arg combined overload. CbA=bfp8, CbB=fp32 is a max
// format delta, so any srca/srcb arg-routing regression fails.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::BinaryFpu<cb_a, cb_b>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
