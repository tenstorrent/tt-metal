// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr auto tiles_per_chunk = get_arg(args::tiles_per_chunk);
    constexpr auto num_chunks = get_arg(args::num_chunks);
    constexpr auto last_chunk_tiles = get_arg(args::last_chunk_tiles);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    // Process the block in chunks to fit within SRAM limits.
    // When num_tiles_per_block divides evenly (last_chunk_tiles ==
    // tiles_per_chunk), use the original single-call path. Otherwise the
    // last chunk is partial and is tilized separately with its own template
    // tile count.
    if constexpr (last_chunk_tiles == tiles_per_chunk) {
        compute_kernel_lib::tilize<
            tiles_per_chunk,
            dfb::in,
            dfb::out,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
            per_core_block_cnt * num_chunks);
    } else {
        for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
            if constexpr (num_chunks > 1) {
                compute_kernel_lib::tilize<
                    tiles_per_chunk,
                    dfb::in,
                    dfb::out,
                    compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                    compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                    compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(num_chunks - 1);
            }
            compute_kernel_lib::tilize<
                last_chunk_tiles,
                dfb::in,
                dfb::out,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
        }
    }
}
