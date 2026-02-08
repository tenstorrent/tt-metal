// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"

namespace NAMESPACE {
void MAIN {
    uint32_t responsibility = get_arg_val<uint32_t>(0);

    uint32_t src0_cb_index = get_compile_time_arg_val(0);
    uint32_t src1_cb_index = get_compile_time_arg_val(1);
    const uint32_t per_core_block_tile_cnt = 1;

    compute_kernel_hw_startup(src0_cb_index, src1_cb_index);
    tilize_init(src0_cb_index, per_core_block_tile_cnt, src1_cb_index);

    for (uint32_t b = 0; b < responsibility; ++b) {
        cb_wait_front(src0_cb_index, per_core_block_tile_cnt);
        cb_reserve_back(src1_cb_index, per_core_block_tile_cnt);

        tilize_block(src0_cb_index, per_core_block_tile_cnt, src1_cb_index);

        cb_push_back(src1_cb_index, per_core_block_tile_cnt);
        cb_pop_front(src0_cb_index, per_core_block_tile_cnt);
    }

    tilize_uninit(src0_cb_index, src1_cb_index);
}
}  // namespace NAMESPACE
