// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    transpose_wh_init(tt::CB::c_intermed0);

    constexpr uint32_t cb_im0 = tt::CB::c_intermed0;
    constexpr uint32_t cb_out1 = tt::CB::c_out1;

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < num_tiles; n++) {
        cb_wait_front(cb_im0, 1);
        cb_reserve_back(cb_out1, 1);

        acquire_dst();
        transpose_wh_tile(cb_im0, 0, 0);
        pack_tile(0, cb_out1);
        release_dst();

        cb_push_back(cb_out1, 1);
        cb_pop_front(cb_im0, 1);
    }
}
}  // namespace NAMESPACE
