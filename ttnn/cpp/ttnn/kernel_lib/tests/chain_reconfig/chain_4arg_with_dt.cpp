// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Case A — 4-arg reconfig_data_format(prev_a, curr_a, prev_b, curr_b) (_with_dt).
//
// Chain shape: BinaryFpu(CbA, CbB) → BinaryFpu(CbC, CbD) → PackTile(CbOut).
// At element 1 the fold sees: srca rotates CbA → CbC (both with prev) AND srcb rotates CbB → CbD (both with prev),
// so emit_pre_element_transitions emits the 4-arg _with_dt overload. The first BinaryFpu's result is overwritten by
// the second; the chain's net semantic is CbC + CbD, packed to CbOut. Argument-order regressions in the 4-arg
// overload surface as wrong outputs when the (CbA, CbB) dtypes differ from (CbC, CbD).

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
        compute_kernel_lib::BinaryFpu<cb_a, cb_b>{},
        compute_kernel_lib::BinaryFpu<cb_c, cb_d>{},
        compute_kernel_lib::PackTile<cb_out>{});
}
