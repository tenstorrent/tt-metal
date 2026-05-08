// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU reduce (no negate). Host sets REDUCE_OP, REDUCE_DIM, REDUCE_FORMAT.
// Similar to reduce.cpp; MIN is dispatched to reduce_sfpu_{h,w}_neg.cpp.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);

    compute_kernel_lib::reduce_sfpu<REDUCE_OP, REDUCE_DIM, REDUCE_FORMAT>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        post_mul_scaler_bits);
}
