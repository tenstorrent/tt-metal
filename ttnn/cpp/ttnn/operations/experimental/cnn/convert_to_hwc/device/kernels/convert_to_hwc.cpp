// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "debug/dprint.h"  // required in all kernels using DPRINT

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"

template <size_t BATCH_SIZE = 1>
FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    pack_untilize_dst_init_short<BATCH_SIZE>(cb_in);

    cb_wait_front(cb_in, BATCH_SIZE);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_init_short(cb_in);
    transpose_wh_tile(cb_in, 0, 0);

    cb_reserve_back(cb_out, BATCH_SIZE);
    pack_untilize_dst<1>(cb_out, BATCH_SIZE);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_out, BATCH_SIZE);
    cb_pop_front(cb_in, BATCH_SIZE);
}

FORCE_INLINE void tilize(
    uint32_t cb_in, uint32_t total_tiles_per_block, uint32_t total_sticks_per_block, uint32_t cb_out) {
    tilize_init_short(cb_in, total_tiles_per_block, cb_out);

    cb_wait_front(cb_in, total_sticks_per_block);
    cb_reserve_back(cb_out, total_tiles_per_block);

    tilize_block(cb_in, total_tiles_per_block, cb_out);

    cb_pop_front(cb_in, total_sticks_per_block);
    cb_push_back(cb_out, total_tiles_per_block);

    tilize_uninit(cb_in, cb_out);
}

namespace NAMESPACE {
void MAIN {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    const uint32_t total_sticks_per_block = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiled_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_transpose_in = get_compile_time_arg_val(2);

    cb_push_back(cb_in, total_sticks_per_block);

    pack_untilize_init(cb_tiled_in, cb_transpose_in);
    transpose_wh_init(cb_tiled_in, cb_transpose_in);
    tilize_init(cb_in, total_tiles, cb_tiled_in);

    tilize(cb_in, total_tiles, total_sticks_per_block, cb_tiled_in);
    for (uint32_t idx = 0; idx < total_tiles; idx++) {
        transpose(cb_tiled_in, cb_transpose_in);
    }
}
}  // namespace NAMESPACE
