// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);

    pack_untilize_init<Wt>(in_cb, untilized_in_cb);

    cb_wait_front(in_cb, Wt);
    cb_reserve_back(untilized_in_cb, Wt);
    pack_untilize_block<Wt>(in_cb, 1, untilized_in_cb);
    cb_push_back(untilized_in_cb, Wt);
    cb_pop_front(in_cb, Wt);

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        reconfig_data_format_srca(in_cb, cache_cb);
        pack_reconfig_data_format(out_cb, untilized_cache_cb);

        pack_untilize_init_short<Wt>(cache_cb, untilized_cache_cb);

        // Untilize a block from the cache
        cb_wait_front(cache_cb, Wt);
        cb_reserve_back(untilized_cache_cb, Wt);

        pack_untilize_block<Wt>(cache_cb, 1, untilized_cache_cb);

        cb_push_back(untilized_cache_cb, Wt);
        cb_pop_front(cache_cb, Wt);

        pack_untilize_uninit(untilized_cache_cb);

        reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
        pack_reconfig_data_format(untilized_cache_cb, out_cb);

        tilize_init_short(untilized_cache2_cb, Wt);

        // Wait on writer to update block. Tilize.
        cb_wait_front(untilized_cache2_cb, Wt);

        cb_reserve_back(out_cb, Wt);

        tilize_block(untilized_cache2_cb, Wt, out_cb);

        tilize_uninit(untilized_cache2_cb);
        cb_push_back(out_cb, Wt);
        cb_pop_front(untilized_cache2_cb, Wt);
    }
}
}  // namespace NAMESPACE
