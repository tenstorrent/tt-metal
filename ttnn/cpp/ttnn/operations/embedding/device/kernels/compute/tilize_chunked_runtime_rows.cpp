// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Chunked tilize for the runtime-rows tilized-output embedding program: each block is tilized as
// num_chunks chunks with a possibly partial last chunk, and the block count is derived from
// args::core_idx (per-core) and the common args::blocks_per_batch.
//
// Binding vocabulary the Metal 2.0 KernelSpec supplies for this source:
//   dfb::in  — row-major weights staging DFB, bound CONSUMER
//   dfb::out — tiled output DFB, bound PRODUCER
//   args::tiles_per_chunk, args::num_chunks, args::last_chunk_tiles, args::batch, args::num_cores — compile-time args

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"
#include "ttnn/operations/embedding/device/kernels/embeddings_fused_split.hpp"

void kernel_main() {
    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t blocks_per_batch = get_arg(args::blocks_per_batch);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);
    const auto [start_block, end_block] = embeddings_fused_block_range(core_idx, blocks_per_batch, batch, num_cores);
    const uint32_t per_core_block_cnt = end_block - start_block;
    // The kernel_lib tilize asserts num_blocks > 0, so a core past the end of a short input returns.
    if (per_core_block_cnt == 0) {
        return;
    }
    constexpr uint32_t tiles_per_chunk = get_arg(args::tiles_per_chunk);
    constexpr uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr uint32_t last_chunk_tiles = get_arg(args::last_chunk_tiles);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    // Process the block in chunks to fit within L1 memory limits.
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
