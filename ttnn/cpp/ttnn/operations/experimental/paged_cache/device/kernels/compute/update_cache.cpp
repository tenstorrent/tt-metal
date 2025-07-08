// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

void MAIN {
    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);

    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(Wt);
    constexpr uint32_t block_ct_dim = Wt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Wt;

    pack_untilize_init<block_ct_dim, full_ct_dim>(in_cb, untilized_in_cb);

    cb_reserve_back(untilized_in_cb, Wt);
    for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
        cb_wait_front(in_cb, block_ct_dim);
        pack_untilize_block<block_ct_dim, full_ct_dim>(in_cb, 1, untilized_in_cb, b);
        cb_pop_front(in_cb, block_ct_dim);
    }
    cb_push_back(untilized_in_cb, Wt);

    reconfig_data_format_srca(in_cb, cache_cb);
    pack_reconfig_data_format(untilized_in_cb, untilized_cache_cb);
    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        pack_untilize_init_short<block_ct_dim, full_ct_dim>(cache_cb, untilized_cache_cb);

        // Untilize a block from the cache
        cb_reserve_back(untilized_cache_cb, Wt);

        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(cache_cb, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cache_cb, 1, untilized_cache_cb, b);
            cb_pop_front(cache_cb, block_ct_dim);
        }

        cb_push_back(untilized_cache_cb, Wt);

        pack_untilize_uninit(untilized_cache_cb);

        reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
        pack_reconfig_data_format(untilized_cache_cb, out_cb);

        tilize_init(untilized_cache2_cb, Wt, out_cb);

        // Wait on writer to update block. Tilize.
        cb_wait_front(untilized_cache2_cb, Wt);

        cb_reserve_back(out_cb, Wt);

        tilize_block(untilized_cache2_cb, Wt, out_cb);

        cb_push_back(out_cb, Wt);
        cb_pop_front(untilized_cache2_cb, Wt);
        tilize_uninit_with_dt(untilized_cache2_cb, cache_cb, out_cb);
        pack_reconfig_data_format(out_cb, untilized_cache_cb);
    }
}
}  // namespace NAMESPACE
