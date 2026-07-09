// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mixed-prev reconfig fallback: at the BinaryFpu, srca has a prev (from CopyTile), srcb is first-emit.
//
// CopyTile(CbA) -> BinaryFpu(CbB,CbC) -> PackTile(CbOut). At element 1 srca rotates CbA->CbB with
// prev set (2-arg _with_dt srca reconfig) while srcb is first-emit CbC (1-arg srcb reconfig). Net
// semantic = CbB+CbC (CopyTile's D0 is overwritten; D0 isn't pipelined between elements without
// DestReuse).

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
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::BinaryFpu<cb_b, cb_c>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
