// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case B — 2-arg combined reconfig_data_format(curr_a, curr_b) (no _with_dt; unconditional reprogram).
//
// Chain shape: BinaryFpu(CbA, CbB) -> PackTile(CbOut).
// Element 0 BinaryFpu is the first chain element, so srca and srcb both have NO_PREV_DFB.
// emit_pre_element_transitions sees reconf_a=true, reconf_b=true, both prev==NO_PREV_DFB and emits the
// 2-arg combined overload at element 0. CbA=bfp8, CbB=fp32 produces max format delta so any srca/srcb
// argument-routing regression surfaces immediately.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        total_tiles, compute_kernel_lib::BinaryFpu<cb_a, cb_b>{}, compute_kernel_lib::PackTile<cb_out>{});
}
