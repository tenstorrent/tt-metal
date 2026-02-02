// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id0 = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_compile_time_arg_val(4);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);

    // Initialize once before the loop
    compute_kernel_lib::untilize_init<tiles_per_row, src_cb_id, out_cb_id0>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // Use unified untilize with Neither mode since we handle init/uninit outside the loop
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither>(block_size);
        }
    }

    // Uninit after loop
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id0>();
}
