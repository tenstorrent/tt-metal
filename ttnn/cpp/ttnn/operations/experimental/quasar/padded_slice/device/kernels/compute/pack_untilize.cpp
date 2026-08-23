// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id0 = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);

    const uint32_t total_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);
    compute_kernel_lib::untilize_init<tiles_per_row, src_cb_id, out_cb_id0>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id0>();
}
