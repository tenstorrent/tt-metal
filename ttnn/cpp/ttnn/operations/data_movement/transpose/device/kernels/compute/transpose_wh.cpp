// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_arg_val<uint32_t>(0);

    transpose_wh_init(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    cb_wait_front(tt::CBIndex::c_0, 32);
    for (uint32_t n = 0; n < NHtWt; n++) {
        // for (uint8_t r = 0; r < 32; ++r) {
        //     SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        //     UNPACK(DPRINT << TSLICE(0, 0, sr, true, true) << ENDL());
        // }
        cb_reserve_back(tt::CBIndex::c_16, 1);

        acquire_dst();
        transpose_wh_tile(tt::CBIndex::c_0, n, 0);
        pack_tile(0, tt::CBIndex::c_16);
        release_dst();
        for (uint8_t r = 0; r < 32; ++r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = uint8_t(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            PACK(DPRINT << TSLICE(16, n, sr, true, true) << ENDL());
        }
        cb_push_back(tt::CBIndex::c_16, 1);
    }
    cb_pop_front(tt::CBIndex::c_0, 32);
}
}  // namespace NAMESPACE
