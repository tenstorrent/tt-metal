// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
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
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(5);
    constexpr uint32_t out_cb = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t num_heads = get_compile_time_arg_val(8);

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    // Untilize input (single block, init only - no uninit needed)
    if (!is_input1) {
        compute_kernel_lib::
            untilize<Wt, in2_cb, untilized_in_cb, compute_kernel_lib::untilize_config::InitUninitMode::InitOnly>(1);
    } else {
        compute_kernel_lib::
            untilize<Wt, in1_cb, untilized_in_cb, compute_kernel_lib::untilize_config::InitUninitMode::InitOnly>(1);
    }

    for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
        // Untilize a block from the cache with reconfiguration from previous iteration
        compute_kernel_lib::untilize<
            Wt,
            cache_cb,
            untilized_cache_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);

        // Wait on writer to update block. Tilize with reconfiguration
        compute_kernel_lib::tilize<
            untilized_cache2_cb,
            out_cb,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Wt, 1);
    }
}
