// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Chunked tilize for the tilized-output embedding program factory: used when a block's tiles do not
// fit the L1 staging budget in one go, so each block is tilized as num_chunks chunks with a possibly
// partial last chunk.
//
// Binding vocabulary the Metal 2.0 KernelSpec supplies for this source:
//   dfb::in  — row-major weights staging DFB, bound CONSUMER
//   dfb::out — tiled output DFB, bound PRODUCER
//   args::per_core_block_cnt, args::tiles_per_chunk, args::num_chunks, args::last_chunk_tiles — compile-time args

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"
#ifdef OUT_SELF_LOOP
#include "api/compute/tile_move_copy.h"  // dummy_unpack
#include "api/dataflow/dataflow_buffer.h"
#endif

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
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
#ifdef OUT_SELF_LOOP
    // Sharded output (embeddings): there is no writer, so this kernel is the output shard's only producer and
    // consumer (self-loop). Pop everything it pushed so the buffer is left balanced; the tiles stay in place.
    // dummy_unpack orders the pop after the wait on Quasar (nothing unpacks this buffer).
    {
        DataflowBuffer out_dfb(dfb::out);
        out_dfb.wait_front(per_core_block_cnt * ((num_chunks - 1) * tiles_per_chunk + last_chunk_tiles));
        dummy_unpack(dfb::out);
        out_dfb.pop_front(per_core_block_cnt * ((num_chunks - 1) * tiles_per_chunk + last_chunk_tiles));
    }
#endif
}
