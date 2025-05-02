// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "debug/dprint.h"  // required in all kernels using DPRINT

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    pack_untilize_dst_init_short<1>(cb_in);

    cb_wait_front(cb_in, ONE_TILE);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_init_short(cb_in);
    transpose_wh_tile(cb_in, 0, 0);

    cb_reserve_back(cb_out, ONE_TILE);
    pack_untilize_dst<1>(cb_out, ONE_TILE);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

template <int BATCH_SIZE>
FORCE_INLINE void tilize(uint32_t cb_in, uint32_t cb_out) {
    tilize_init_short(cb_in, BATCH_SIZE, cb_out);

    cb_wait_front(cb_in, BATCH_SIZE);
    cb_reserve_back(cb_out, BATCH_SIZE);

    tilize_block(cb_in, BATCH_SIZE, cb_out);

    cb_pop_front(cb_in, BATCH_SIZE);
    cb_push_back(cb_out, BATCH_SIZE);

    tilize_uninit(cb_in, cb_out);
}

namespace NAMESPACE {
void MAIN {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiled_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_transpose_in = get_compile_time_arg_val(2);

    pack_untilize_init(cb_tiled_in, cb_transpose_in);
    transpose_wh_init(cb_tiled_in, cb_transpose_in);
    tilize_init(cb_in, 1, cb_tiled_in);

    for (uint32_t idx = 0; idx < total_tiles; idx++) {
        tilize<1>(cb_in, cb_tiled_in);
        transpose(cb_tiled_in, cb_transpose_in);
    }
}
}  // namespace NAMESPACE
