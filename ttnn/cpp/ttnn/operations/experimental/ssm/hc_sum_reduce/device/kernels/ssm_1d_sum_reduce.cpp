// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_COL

#include <cstdint>

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/transpose_wh.h"

constexpr uint32_t ONE_TILE = 1;

FORCE_INLINE void transpose(uint32_t cb_in, uint32_t cb_out) {
    cb_wait_front(cb_in, ONE_TILE);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_wh_init_short(cb_in);
    transpose_wh_tile(cb_in, 0, 0);

    cb_reserve_back(cb_out, ONE_TILE);
    pack_tile(0, cb_out);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

FORCE_INLINE void reduce(uint32_t cb_in, uint32_t cb_scalar, uint32_t cb_out) {
    cb_wait_front(cb_in, ONE_TILE);

    tile_regs_acquire();
    tile_regs_wait();

    reduce_init_delta<false, REDUCE_OP, REDUCE_DIM>(cb_in, cb_scalar, cb_out);
    reduce_tile(cb_in, cb_scalar, 0, 0, 0);
    reduce_revert_delta<REDUCE_DIM>(cb_out);

    cb_reserve_back(cb_out, ONE_TILE);
    pack_tile(0, cb_out);

    tile_regs_commit();
    tile_regs_release();

    cb_push_back(cb_out, ONE_TILE);
    cb_pop_front(cb_in, ONE_TILE);
}

namespace NAMESPACE {
void MAIN {
    uint32_t num_blocks = get_arg_val<uint32_t>(0);
    uint32_t input_num_blocks_h = get_arg_val<uint32_t>(1);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t intermed_cb_id0 = get_compile_time_arg_val(2);
    constexpr uint32_t intermed_cb_id1 = get_compile_time_arg_val(3);
    constexpr uint32_t intermed_cb_id2 = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(5);

    reduce_init<true>(input_cb_id, scalar_cb_id);
    reduce_revert_delta<REDUCE_DIM>(intermed_cb_id1);  // Required or else the first tile is wrong

    for(uint32_t block_h_id = 0; block_h_id < input_num_blocks_h; block_h_id++){

        cb_wait_front(scalar_cb_id, ONE_TILE);

        for (uint32_t output_idx = 0; output_idx < num_blocks; output_idx++) {
            for (uint32_t slice_idx = 0; slice_idx < TILE_WIDTH; slice_idx++) {
                reconfig_data_format_srca(intermed_cb_id2, input_cb_id);
                pack_reconfig_data_format(output_cb_id, intermed_cb_id0);
                transpose(input_cb_id, intermed_cb_id0);  // 32 x B
                reconfig_data_format_srca(input_cb_id, intermed_cb_id0);
                reduce(intermed_cb_id0, scalar_cb_id, intermed_cb_id1);  // 1 x B
            }
            // Get full tile back from writer and transpose it
            pack_reconfig_data_format(intermed_cb_id0, output_cb_id);
            transpose(intermed_cb_id2, output_cb_id);
        }
    }
}
}  // namespace NAMESPACE
