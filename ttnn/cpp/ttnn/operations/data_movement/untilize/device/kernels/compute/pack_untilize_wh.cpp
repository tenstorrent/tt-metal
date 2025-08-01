// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "debug/dprint.h"
#include "common.cpp"

namespace NAMESPACE {
void MAIN {
#ifdef DST_ACCUM_MODE
    constexpr uint32_t max_bct = 4;
#else
    constexpr uint32_t max_bct = 8;
#endif
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_column(block_size_row, max_bct);
    constexpr uint32_t block_ct_dim = block_size_row / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = block_size_row;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    pack_untilize_init<block_ct_dim, full_ct_dim>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        cb_reserve_back(tt::CBIndex::c_16, full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(tt::CBIndex::c_0, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(tt::CBIndex::c_0, 1, tt::CBIndex::c_16, b);
            cb_pop_front(tt::CBIndex::c_0, block_ct_dim);
        }
        cb_push_back(tt::CBIndex::c_16, full_ct_dim);
    }
    pack_untilize_uninit(tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
