// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);
    // Each region binds the buffer set matching its block width, so the indices are compile-time
    // args rather than a fixed c_0/c_16 -- see BlockBufferSet in data_movement/common.
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(cb_id_in, cb_id_out);
    compute_kernel_lib::untilize<
        block_size_row,
        cb_id_in,
        cb_id_out,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
        block_size_col * third_dim);
}
