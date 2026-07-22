// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Typecast

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);

    init_sfpu(input_cb, output_cb);

    constexpr uint32_t total_tiles = per_core_block_cnt * per_core_block_dim;
    compute_kernel_lib::unary<
        compute_kernel_lib::Typecast<CHAIN_TYPECAST_IN_DF, CHAIN_TYPECAST_OUT_DF, compute_kernel_lib::Dst::D0>,
        compute_kernel_lib::input(
            input_cb, compute_kernel_lib::InputLifecycle::Streaming, compute_kernel_lib::DataFormatReconfig::Disabled),
        compute_kernel_lib::output(
            output_cb,
            compute_kernel_lib::OutputLifecycle::Streaming,
            compute_kernel_lib::DataFormatReconfig::Disabled)>(compute_kernel_lib::EltwiseShape::tiles(total_tiles));
}
