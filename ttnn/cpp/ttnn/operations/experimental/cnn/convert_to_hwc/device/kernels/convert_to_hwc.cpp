// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"

template <uint32_t BatchSize = 1>
FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, BatchSize);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_init_short(cb_in);
    for (uint32_t i = 0; i < BatchSize; i++) {
        transpose_wh_tile(cb_in, i, i);
    }

    cb_reserve_back(cb_out, BatchSize);
    pack_untilize_dst<1>(cb_out, BatchSize);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_out, BatchSize);
    cb_pop_front(cb_in, BatchSize);
}

FORCE_INLINE void tilize(
    uint32_t cb_in, uint32_t total_tiles_per_block, uint32_t total_sticks_per_block, uint32_t cb_out) {
    cb_wait_front(cb_in, total_sticks_per_block);
    cb_reserve_back(cb_out, total_tiles_per_block);

    tilize_block(cb_in, total_tiles_per_block, cb_out);

    cb_pop_front(cb_in, total_sticks_per_block);
    cb_push_back(cb_out, total_tiles_per_block);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiled_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_transpose_in0 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_transpose_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t total_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t total_sticks_per_block = get_compile_time_arg_val(5);
    constexpr uint32_t is_input_in_dram = get_compile_time_arg_val(6);

    compute_kernel_hw_startup(cb_in, cb_tiled_in);
    if constexpr (!is_input_in_dram) {
        cb_push_back(cb_in, total_sticks_per_block);
    }

    tilize_init(cb_in, total_tiles, cb_tiled_in);
    tilize(cb_in, total_tiles, total_sticks_per_block, cb_tiled_in);
    tilize_uninit(cb_in, cb_tiled_in);

    pack_untilize_init(cb_in, cb_transpose_in0);
    transpose_wh_init(cb_in, cb_transpose_in0);
    pack_untilize_dst_init_short<1>(cb_in);

    for (uint32_t idx = 0; idx < total_tiles; idx++) {
        const uint32_t cb_transpose_in = idx % 2 == 0 ? cb_transpose_in0 : cb_transpose_in1;
        transpose<1>(cb_tiled_in, cb_transpose_in);
    }
    pack_untilize_uninit(cb_transpose_in0);
}
}  // namespace NAMESPACE
