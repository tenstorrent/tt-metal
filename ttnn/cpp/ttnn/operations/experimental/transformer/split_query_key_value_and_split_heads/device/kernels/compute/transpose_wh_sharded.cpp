// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(tt::CBIndex::c_24, tt::CBIndex::c_17);
    transpose_init(tt::CBIndex::c_24);

    constexpr uint32_t cb_im0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < num_tiles; n++) {
        cb_wait_front(cb_im0, 1);
        cb_reserve_back(cb_out1, 1);

        acquire_dst();
        transpose_tile(cb_im0, 0, 0);
        pack_tile(0, cb_out1);
        release_dst();

        cb_push_back(cb_out1, 1);
        cb_pop_front(cb_im0, 1);
    }
}
}  // namespace NAMESPACE
