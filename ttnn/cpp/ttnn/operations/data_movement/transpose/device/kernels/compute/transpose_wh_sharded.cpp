// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_arg_val<uint32_t>(0);
    uint32_t HtWt = get_arg_val<uint32_t>(1);
    uint32_t N = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);

    transpose_wh_init(cb_id_in);

    // transpose a row-major block:
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile

    uint32_t tile_idx = 0;
    uint32_t tile_idx_N = 0;

    cb_wait_front(cb_id_in, NHtWt);
    cb_reserve_back(cb_id_out, NHtWt);
    for (uint32_t n = 0; n < N; ++n) {
        tile_idx = tile_idx_N;
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t h = 0; h < Ht; ++h) {
                tile_regs_acquire();
                transpose_wh_tile(cb_id_in, tile_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_id_out);
                tile_regs_release();
                tile_idx += Wt;
            }
            tile_idx = tile_idx - HtWt + 1;
        }
        tile_idx_N += HtWt;
    }
    cb_push_back(cb_id_out, NHtWt);
    cb_pop_front(cb_id_in, NHtWt);
}
}  // namespace NAMESPACE
