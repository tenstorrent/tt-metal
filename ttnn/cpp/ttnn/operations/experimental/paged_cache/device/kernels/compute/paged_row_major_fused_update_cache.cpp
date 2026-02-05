// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    uint32_t rt_args_idx = 0;
    const bool has_work = get_arg_val<uint32_t>(rt_args_idx++);
    if (!has_work) {
        return;
    }
    const bool is_input1 = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t in1_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in2_cb = get_compile_time_arg_val(1);
    uint32_t in_cb = in1_cb;
    if (!is_input1) {
        in_cb = in2_cb;
    }

    constexpr uint32_t cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t Wt = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);

    compute_kernel_hw_startup(cache_cb, untilized_cache_cb);

    // Track previous CBs for reconfiguration in loop
    // First iteration uses hw_startup CBs as previous
    uint32_t prev_cb_srca = cache_cb;
    uint32_t prev_cb_output = untilized_cache_cb;

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration
        compute_kernel_lib::untilize<
            Wt,
            cache_cb,
            untilized_cache_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::Reconfigure>(
            1, compute_kernel_lib::untilize_config::PreviousCBs{prev_cb_srca, prev_cb_output});

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<
            untilized_cache2_cb,
            out_cb,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::TilizeSpeedMode::Standard,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::Reconfigure>(
            Wt,
            1,
            compute_kernel_lib::tilize_config::NonTileAlignedCBWaitConfig::disabled(),
            compute_kernel_lib::tilize_config::PreviousCBs{cache_cb, untilized_cache_cb});

        // Update previous CBs for next iteration
        prev_cb_srca = untilized_cache2_cb;
        prev_cb_output = out_cb;
    }
}
