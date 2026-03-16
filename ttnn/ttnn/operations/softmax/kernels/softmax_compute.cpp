// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

namespace NAMESPACE {

void MAIN {
    // --- Compile-time args ---
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);      // 0
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);     // 8
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);        // 16
    constexpr uint32_t cb_max = get_compile_time_arg_val(3);        // 24
    constexpr uint32_t cb_exp = get_compile_time_arg_val(4);        // 25
    constexpr uint32_t cb_recip_sum = get_compile_time_arg_val(5);  // 26
    constexpr uint32_t R = get_compile_time_arg_val(6);             // tiles per work unit
    constexpr uint32_t numeric_stable = get_compile_time_arg_val(7);
    constexpr uint32_t num_work_units = get_compile_time_arg_val(8);

    // Stage 2: exp_only — copy with exp post-op
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_out);

    auto exp_post_op = [](uint32_t dst_idx) {
        exp_tile_init();
        exp_tile(dst_idx);
    };

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        compute_kernel_lib::copy_tiles<
            compute_kernel_lib::CopyInputPolicy::WaitAndPop,
            compute_kernel_lib::CopyDataFormatReconfig::NONE>(cb_input, cb_out, R, exp_post_op);
    }
}

}  // namespace NAMESPACE
