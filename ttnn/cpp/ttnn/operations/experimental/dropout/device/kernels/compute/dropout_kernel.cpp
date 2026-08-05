// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/dropout.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    uint32_t int_probability = get_compile_time_arg_val(2);
    uint32_t int_scale_factor = get_compile_time_arg_val(3);

    uint32_t seed = get_arg_val<uint32_t>(0);

    CircularBuffer cb_in(tt::CBIndex::c_0);
    CircularBuffer cb_out(tt::CBIndex::c_2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2);
    copy_init(tt::CBIndex::c_0);
    dropout_kernel_init(seed);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_in.wait_front(1);

            copy_tile(tt::CBIndex::c_0, 0, 0);

            dropout_tile(0, int_probability, int_scale_factor);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, tt::CBIndex::c_2);

            cb_in.pop_front(1);

            tile_regs_release();
        }
        cb_out.push_back(per_core_block_dim);
    }
}
