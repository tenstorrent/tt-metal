// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case D — single-side _with_dt on srca. srca rotates across CBs of different dtypes; srcb untouched.
//
// Chain shape: CopyTile(CbA, D0) -> CopyTile(CbB, D0) -> PackTile(CbOut).
// CopyTile only touches srca. At element 1: prev_a=CbA, curr_a=CbB, prev set → reconf_a=true,
// emits reconfig_data_format_srca(prev_a, curr_a) (2-arg per-side _with_dt). srcb stays NO_PREV_DFB
// throughout, so no srcb emission. Pack stays the same dtype across both PackTile-less intermediate
// CopyTiles; pack reconfig fires only at the final PackTile (first-emit single-arg).
//
// CbA=bfp8, CbB=bf16 forces the srca-side rotation to span block-float -> IEEE — strongest single-side
// format delta. Net semantic = CbB (the second CopyTile overwrites D0).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        total_tiles,
        compute_kernel_lib::CopyTile<cb_a>{},
        compute_kernel_lib::CopyTile<cb_b>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
