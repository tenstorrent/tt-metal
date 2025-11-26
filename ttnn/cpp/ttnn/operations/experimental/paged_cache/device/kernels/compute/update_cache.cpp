// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.h"

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

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    // Untilize input (single row, no uninit needed)
    compute_kernel_lib::untilize<Wt, true, false>(in_cb, untilized_in_cb, 1);

    reconfig_data_format_srca(in_cb, cache_cb);
    pack_reconfig_data_format(untilized_in_cb, untilized_cache_cb);
    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache - DEST limit auto-detected
        compute_kernel_lib::untilize<Wt>(cache_cb, untilized_cache_cb, 1);

        reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
        pack_reconfig_data_format(untilized_cache_cb, out_cb);

        // Wait on writer to update block. Tilize.
        compute_kernel_lib::tilize<true, true, false, true>(
            untilized_cache2_cb,  // new_cb (input)
            Wt,                   // block_w
            out_cb,               // output CB
            1,                    // num_blocks (1 iteration)
            1,                    // subblock_h (default)
            cache_cb              // old_cb (for DT restoration)
        );

        pack_reconfig_data_format(out_cb, untilized_cache_cb);
    }
}
}  // namespace NAMESPACE
