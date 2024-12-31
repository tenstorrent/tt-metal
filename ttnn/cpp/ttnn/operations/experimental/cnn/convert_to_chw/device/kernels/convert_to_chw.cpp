// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/transpose_wh.h"

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    transpose_wh_init_short(cb_in);

    pack_untilize_dst_init_short<ONE_TILE, 1, false, false, 32>(cb_out);

    cb_wait_front(cb_in, ONE_TILE);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_tile(cb_in, 0, 0);

    cb_reserve_back(cb_out, ONE_TILE);

    pack_untilize_dst<ONE_TILE>(cb_out);

    tile_regs_commit();
    tile_regs_release();

    pack_untilize_uninit();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

namespace NAMESPACE {
void MAIN {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_transpose_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    transpose_wh_init(cb_in, cb_transpose_in);
    pack_untilize_init(cb_in, cb_transpose_in);

    for (uint32_t idx = 0; idx < total_tiles; idx++) {
        transpose(cb_in, cb_transpose_in);
    }
}
}  // namespace NAMESPACE
