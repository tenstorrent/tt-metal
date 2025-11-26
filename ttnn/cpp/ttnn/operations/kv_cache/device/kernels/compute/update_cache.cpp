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
    constexpr uint32_t num_batched_heads = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t granularity = get_compile_time_arg_val(8);
    constexpr uint32_t u_count = get_compile_time_arg_val(9);

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    // Initialize once before the loop
    compute_kernel_lib::untilize_init<Wt>(in_cb, untilized_in_cb);

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        // Untilize input (init done before loop, no uninit needed)
        compute_kernel_lib::untilize<Wt, false, false>(in_cb, untilized_in_cb, 1);

        reconfig_data_format_srca(in_cb, cache_cb);
        for (uint32_t u = 0; u < u_count; ++u) {
            // Untilize cache blocks
            compute_kernel_lib::untilize<Wt>(cache_cb, untilized_cache_cb, granularity);

            reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
            pack_reconfig_data_format(untilized_cache_cb, out_cb);

            // Wait on writer to update block. Tilize.
            compute_kernel_lib::tilize<true, true, false, true>(
                untilized_cache2_cb,  // new_cb (input)
                Wt,                   // block_w
                out_cb,               // output CB
                granularity,          // num_blocks
                1,                    // subblock_h (default)
                cache_cb              // old_cb (for DT restoration)
            );

            pack_reconfig_data_format(out_cb, untilized_cache_cb);
        }
        reconfig_data_format_srca(cache_cb, in_cb);

        // Re-initialize for next iteration
        compute_kernel_lib::untilize_init<Wt>(in_cb, untilized_in_cb);
    }
}
}  // namespace NAMESPACE
