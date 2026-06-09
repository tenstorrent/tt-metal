// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case C — mixed-prev fallback. srca has prev (from CopyTile), srcb is first-emit.
//
// Chain shape: CopyTile(CbA, D0) -> BinaryFpu(CbB, CbC, D0) -> PackTile(CbOut).
// At BinaryFpu (element 1): prev_a=CbA (set by element 0's CopyTile), curr_a=CbB → reconf_a=true,
// prev_a known. prev_b=NO_PREV_DFB (element 0's CopyTile doesn't touch srcb), curr_b=CbC →
// reconf_b=true, prev_b first-emit. emit_pre_element_transitions takes the mixed-prev branch:
// reconfig_data_format_srca(prev_a, curr_a) (2-arg _with_dt) plus reconfig_data_format_srcb(curr_b) (1-arg).
//
// Note: BinaryFpu reads srca from CbB (not CbA) — CopyTile's result in D0 is overwritten because the
// chain library doesn't pipeline D0 between elements unless DestReuse is used. Net semantic = CbB + CbC.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        total_tiles,
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::BinaryFpu<cb_b, cb_c>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
