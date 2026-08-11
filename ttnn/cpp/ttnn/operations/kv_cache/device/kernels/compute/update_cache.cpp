// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
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

    // Per-core: Bcache and the batch index at the start of this core's work. users_this_tile is
    // derived each group as min(32, Bcache - b) so the last group of a non-tile batch is short.
    const uint32_t Bcache = get_arg_val<uint32_t>(0);
    const uint32_t batch_start_id = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    uint32_t b = batch_start_id;
    for (uint32_t h = 0; h < num_batched_heads; ++h) {
        // Untilize input (standalone operation)
        compute_kernel_lib::untilize<
            Wt,
            in_cb,
            untilized_in_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

        uint32_t users_remaining = std::min(32u, Bcache - b);
        while (users_remaining > 0) {
            const uint32_t g = std::min(granularity, users_remaining);
            compute_kernel_lib::untilize<Wt, cache_cb, untilized_cache_cb>(g);

            // Wait on writer to update block, then tilize back
            compute_kernel_lib::tilize<Wt, untilized_cache2_cb, out_cb>(g);

            // Keep b in sync with reader/writer so the next group's user count is correct.
            for (uint32_t i = 0; i < g; ++i) {
                b++;
                if (b == Bcache) {
                    b = 0;
                }
            }
            users_remaining -= g;
        }
        reconfig_data_format_srca(cache_cb, in_cb);
    }
}
