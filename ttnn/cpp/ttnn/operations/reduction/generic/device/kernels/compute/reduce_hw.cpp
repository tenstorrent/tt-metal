// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb3(tt::CBIndex::c_3);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    // REDUCE_OP/DIM is expected to come from add_define
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
        tt::CBIndex::c_0,  // input CB
        tt::CBIndex::c_2,  // scaler CB
        tt::CBIndex::c_3,  // output CB
        Ht,
        Wt,
        NC);
}
