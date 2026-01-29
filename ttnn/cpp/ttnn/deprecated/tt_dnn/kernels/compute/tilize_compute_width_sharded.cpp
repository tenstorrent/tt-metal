// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

// #include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    DPRINT << "[COMPUTE] kernel started" << ENDL();
    uint32_t responsibility = get_arg_val<uint32_t>(0);
    // DPRINT << "compute kernel responsibility: " << responsibility << ENDL();

    // uint32_t per_core_block_tile_cnt = get_runtime_time_arg_val(1);
    uint32_t src0_cb_index = get_compile_time_arg_val(0);
    uint32_t src1_cb_index = get_compile_time_arg_val(1);
    const uint32_t per_core_block_tile_cnt = 1;

    compute_kernel_hw_startup(src0_cb_index, src1_cb_index);
    tilize_init(src0_cb_index, per_core_block_tile_cnt, src1_cb_index);

    for (uint32_t b = 0; b < responsibility; ++b) {
        DPRINT << "[COMPUTE] entered for loop, index: " << b << ENDL();
        cb_wait_front(src0_cb_index, per_core_block_tile_cnt);
        cb_reserve_back(src1_cb_index, per_core_block_tile_cnt);
        DPRINT << "[COMPUTE] wait & reserve complete" << ENDL();
        tilize_block(src0_cb_index, per_core_block_tile_cnt, src1_cb_index);  // hang occurs here starting from index 4
        DPRINT << "[COMPUTE] tilize complete" << ENDL();
        cb_push_back(src1_cb_index, per_core_block_tile_cnt);
        cb_pop_front(src0_cb_index, per_core_block_tile_cnt);
        DPRINT << "[COMPUTE] push back & pop front complete" << ENDL();
    }
    tilize_uninit(src0_cb_index, src1_cb_index);
    DPRINT << "[COMPUTE] kernel ended" << ENDL();
}
}  // namespace NAMESPACE
