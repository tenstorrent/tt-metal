// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
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

    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        // Untilize input (standalone operation)
        compute_kernel_lib::untilize<Wt, in_cb, untilized_in_cb>(1);

        reconfig_data_format_srca(in_cb, cache_cb);
        for (uint32_t u = 0; u < u_count; ++u) {
            // Untilize cache blocks
            compute_kernel_lib::untilize<Wt, cache_cb, untilized_cache_cb>(granularity);

            reconfig_data_format_srca(cache_cb, untilized_cache2_cb);
            pack_reconfig_data_format(untilized_cache_cb, out_cb);

            // Wait on writer to update block. Tilize with DT reconfiguration.
            compute_kernel_lib::tilize<
                untilized_cache2_cb,  // input_cb
                out_cb,               // output_cb
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::TilizeSpeedMode::Standard,
                cache_cb>(     // reconfig_from_cb (for DT restoration)
                Wt,            // block_width_tiles
                granularity);  // num_blocks

            pack_reconfig_data_format(out_cb, untilized_cache_cb);
        }
        reconfig_data_format_srca(cache_cb, in_cb);
    }
}
