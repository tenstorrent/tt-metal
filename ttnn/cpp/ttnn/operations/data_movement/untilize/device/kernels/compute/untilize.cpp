// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    untilize_init(src_cb_id);

    DPRINT << "UNTILIZE src_cb_id: " << src_cb_id << ", out_cb_id: " << out_cb_id << ENDL();

    DPRINT << "UNTILIZE per_core_block_cnt: " << per_core_block_cnt << ENDL();
    DPRINT << "UNTILIZE per_core_block_tile_cnt: " << per_core_block_tile_cnt << ENDL();
    DPRINT << "UNTILIZE src_cb_id: " << src_cb_id << ", out_cb_id: " << out_cb_id << ENDL();

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        DPRINT << "UNTILIZE waiting for block " << b << ENDL();
        cb_wait_front(src_cb_id, per_core_block_tile_cnt);
        DPRINT << "UNTILIZE reserving block " << b << ENDL();
        cb_reserve_back(out_cb_id, per_core_block_tile_cnt);

        untilize_block(src_cb_id, per_core_block_tile_cnt, out_cb_id);

        DPRINT << "UNTILIZE pushing block " << b << ENDL();
        PACK(tt::compute::common::print_full_tile(out_cb_id, 2));
        cb_push_back(out_cb_id, per_core_block_tile_cnt);
        DPRINT << "UNTILIZE popping block " << b << ENDL();
        cb_pop_front(src_cb_id, per_core_block_tile_cnt);
    }
}
}  // namespace NAMESPACE
