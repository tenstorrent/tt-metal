// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 4-arg reconfig_data_format(prev_a, curr_a, prev_b, curr_b) (_with_dt).
//
// BinaryFpu(CbA,CbB) -> BinaryFpu(CbC,CbD) -> PackTile(CbOut). At element 1 both srca (CbA->CbC)
// and srcb (CbB->CbD) rotate with prev set, so the chain emits the 4-arg _with_dt overload. Net
// semantic is CbC+CbD (first add overwritten). Arg-order regressions fail when (CbA,CbB) dtypes
// differ from (CbC,CbD).

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_2;
    constexpr uint32_t cb_d = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    compute_kernel_lib::eltwise_chain(
        compute_kernel_lib::EltwiseShape::tiles(total_tiles),
        compute_kernel_lib::BinaryFpu<compute_kernel_lib::input(cb_a), compute_kernel_lib::input(cb_b)>{},
        compute_kernel_lib::BinaryFpu<compute_kernel_lib::input(cb_c), compute_kernel_lib::input(cb_d)>{},
        compute_kernel_lib::PackTile<compute_kernel_lib::output(cb_out)>{});
}
