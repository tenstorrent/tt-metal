// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
// #include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t block_size_col = get_compile_time_arg_val(0);
    constexpr uint32_t block_size_row = get_compile_time_arg_val(1);
    constexpr uint32_t third_dim = get_compile_time_arg_val(2);
    // Compile-time rather than fixed indices: a factory whose work split gives cores different
    // block widths declares one correctly-sized buffer pair per width, and each instance of this
    // kernel binds the pair matching the width its cores were given.
    constexpr uint32_t dfb_id_in = get_compile_time_arg_val(3);
    constexpr uint32_t dfb_id_out = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(dfb_id_in, dfb_id_out);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<dfb_id_in>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        block_size_row,
        dfb_id_in,
        dfb_id_out,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(block_size_col * third_dim);
}
