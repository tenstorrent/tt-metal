// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/transpose_wh.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

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

void kernel_main() {
    uint32_t num_blocks = get_arg_val<uint32_t>(0);
    uint32_t input_num_blocks_h = get_arg_val<uint32_t>(1);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t intermed_cb_id0 = get_compile_time_arg_val(2);
    constexpr uint32_t intermed_cb_id1 = get_compile_time_arg_val(3);
    constexpr uint32_t intermed_cb_id2 = get_compile_time_arg_val(4);
    constexpr uint32_t output_cb_id = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(input_cb_id, scalar_cb_id, intermed_cb_id1);

    for (uint32_t block_h_id = 0; block_h_id < input_num_blocks_h; block_h_id++) {
        cb_wait_front(scalar_cb_id, ONE_TILE);

        for (uint32_t output_idx = 0; output_idx < num_blocks; output_idx++) {
            for (uint32_t slice_idx = 0; slice_idx < TILE_WIDTH; slice_idx++) {
                reconfig_data_format_srca(intermed_cb_id2, input_cb_id);
                pack_reconfig_data_format(output_cb_id, intermed_cb_id0);
                transpose(input_cb_id, intermed_cb_id0);  // 32 x B
                reconfig_data_format_srca(input_cb_id, intermed_cb_id0);
                compute_kernel_lib::reduce<
                    PoolType::SUM,
                    ReduceDim::REDUCE_COL,
                    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                    intermed_cb_id0,
                    scalar_cb_id,
                    intermed_cb_id1,
                    compute_kernel_lib::ReduceInputBlockShape::single());  // 1 x B
            }
            // Get full tile back from writer and transpose it
            pack_reconfig_data_format(intermed_cb_id0, output_cb_id);
            transpose(intermed_cb_id2, output_cb_id);
        }
    }
}
