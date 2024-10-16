// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_compile_time_arg_val(0);
    transpose_wh_init(tt::CB::c_in0);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(tt::CB::c_in0, 1);
        cb_reserve_back(tt::CB::c_out0, 1);

        acquire_dst();
        transpose_wh_tile(tt::CB::c_in0, 0, 0);
        pack_tile(0, tt::CB::c_out0);
        release_dst();

        cb_push_back(tt::CB::c_out0, 1);
        cb_pop_front(tt::CB::c_in0, 1);
    }
}
}  // namespace NAMESPACE
