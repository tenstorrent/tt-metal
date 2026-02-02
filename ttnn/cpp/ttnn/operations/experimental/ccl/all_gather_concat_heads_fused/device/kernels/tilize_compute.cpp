// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// #include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_idx = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out_idx = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_in_idx, cb_out_idx);
    compute_kernel_lib::tilize<cb_in_idx, cb_out_idx>(per_core_block_tile_cnt, per_core_block_cnt);
}
