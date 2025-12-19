// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.h"

namespace NAMESPACE {
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    uint32_t row_chunk = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    // REDUCE_OP/DIM is expected to come from add_define
    // tiles are expected to come in the N C W_skip H W_chunk order
    // row_chunk is provided by the host to match the tile data layout
    compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM>(
        tt::CBIndex::c_0,  // input CB
        tt::CBIndex::c_2,  // scaler CB
        tt::CBIndex::c_3,  // output CB
        Ht,
        Wt,
        NC,
        row_chunk);  // host-provided chunk size for REDUCE_COL
}
}  // namespace NAMESPACE
