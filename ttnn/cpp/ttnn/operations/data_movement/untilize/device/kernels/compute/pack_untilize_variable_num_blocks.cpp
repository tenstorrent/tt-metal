// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);

    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(2);

    pack_untilize_init<per_core_block_tile_cnt>(src_cb_id, out_cb_id);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(src_cb_id, per_core_block_tile_cnt);
        cb_reserve_back(out_cb_id, per_core_block_tile_cnt);

        pack_untilize_block<per_core_block_tile_cnt>(src_cb_id, 1, out_cb_id);

        cb_push_back(out_cb_id, per_core_block_tile_cnt);
        cb_pop_front(src_cb_id, per_core_block_tile_cnt);
    }

    pack_untilize_uninit(out_cb_id);
}
}  // namespace NAMESPACE
