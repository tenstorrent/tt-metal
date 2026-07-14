// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Single-side _with_dt on srca: srca rotates across dtypes; srcb untouched.
//
// CopyTile(CbA) -> CopyTile(CbB) -> PackTile(CbOut). CopyTile only touches srca, so at element 1
// srca rotates CbA->CbB with prev set (2-arg srca _with_dt) and srcb never emits. CbA=bfp8, CbB=bf16
// spans block-float -> IEEE, the strongest single-side delta. Net semantic = CbB.

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
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::CopyTile<cb_b>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
