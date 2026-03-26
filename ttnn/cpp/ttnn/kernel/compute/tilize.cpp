// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"

#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"

// Provide stub functions if debug printing is not enabled
#if defined(COMPILE_FOR_TRISC) && (!defined(DEBUG_PRINT_ENABLED) || defined(FORCE_DPRINT_OFF))
namespace tt::compute::common {
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    // No-op when debug printing is disabled
}
}  // namespace tt::compute::common
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
        UNPACK(tt::compute::common::print_full_tile(tt::CBIndex::c_0, 0));
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        tilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
        UNPACK(tt::compute::common::print_full_tile(tt::CBIndex::c_16, 0, true));
        cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
    }
}
