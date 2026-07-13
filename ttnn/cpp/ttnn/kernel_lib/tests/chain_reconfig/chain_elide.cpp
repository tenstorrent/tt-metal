// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time reconfig elision: same CB on srca across consecutive elements.
//
// CopyTile(CbA) x3 -> PackTile(CbOut). Past element 0, curr_a == prev_a == CbA so reconf_a is
// false at compile time and no srca reconfig is emitted; the pack is the only emission point.
// Net semantic = CbA. Verifies the fold preserves the elision path.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_a, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
