// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t last_chunk_tiles = get_compile_time_arg_val(5);

    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    // Process the block in chunks to fit within L1 memory limits.
    // When num_tiles_per_block divides evenly (last_chunk_tiles ==
    // tiles_per_chunk), use the original single-call path. Otherwise the
    // last chunk is partial and is tilized separately with its own template
    // tile count.
    if constexpr (last_chunk_tiles == tiles_per_chunk) {
        compute_kernel_lib::tilize<
            tiles_per_chunk,
            cb_id_in0,
            cb_id_out0,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
            per_core_block_cnt * num_chunks);
    } else {
        for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
            if constexpr (num_chunks > 1) {
                compute_kernel_lib::tilize<
                    tiles_per_chunk,
                    cb_id_in0,
                    cb_id_out0,
                    compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(num_chunks - 1);
            }
            compute_kernel_lib::tilize<
                last_chunk_tiles,
                cb_id_in0,
                cb_id_out0,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        }
    }
}
