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
        compute_kernel_lib::untilize<
            Wt,
            in_cb,
            untilized_in_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

        // Track previous CBs for inner loop reconfiguration
        uint32_t prev_cb_srca = in_cb;
        uint32_t prev_cb_output = untilized_in_cb;

        for (uint32_t u = 0; u < u_count; ++u) {
            // Untilize cache blocks with reconfiguration
            compute_kernel_lib::untilize<Wt, cache_cb, untilized_cache_cb>(granularity);

            // Wait on writer to update block. Tilize with reconfiguration
            compute_kernel_lib::tilize<Wt, untilized_cache2_cb, out_cb>(granularity  // num_blocks
            );
        }
        reconfig_data_format_srca(cache_cb, in_cb);
    }
}
