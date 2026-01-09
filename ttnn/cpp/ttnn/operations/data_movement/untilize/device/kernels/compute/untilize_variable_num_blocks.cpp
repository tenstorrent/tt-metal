// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// //
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "api/debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);

    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    untilize_init(src_cb_id);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        DPRINT << "START processing block " << b << ENDL();
        DPRINT << "before wait_front" << ENDL();
        cb_wait_front(src_cb_id, per_core_block_tile_cnt);
        DPRINT << "after wait_front" << ENDL();
        cb_reserve_back(out_cb_id, per_core_block_tile_cnt);
        DPRINT << "after reserve_back" << ENDL();

        untilize_block(src_cb_id, per_core_block_tile_cnt, out_cb_id);
        DPRINT << "after untilize_block" << ENDL();
        cb_push_back(out_cb_id, per_core_block_tile_cnt);
        DPRINT << "after push_back" << ENDL();
        cb_pop_front(src_cb_id, per_core_block_tile_cnt);
        DPRINT << "after pop_front" << ENDL();
        DPRINT << "done processing block " << b << ENDL();
    }
    DPRINT << "done processing all blocks" << ENDL();
}
}  // namespace NAMESPACE
