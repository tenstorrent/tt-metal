// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer: stores this cluster's finished blocks of C.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N]. A block is the per_core_M_tiles x per_core_N_tiles tiles of C at
// origin (block_M_tile, block_N_tile) in one batch. This cluster starts at (first_batch, first_M_tile,
// first_N_tile) and writes num_blocks of them, stepping per_core_N_tiles tiles across N, then per_core_M_tiles tiles
// down M, then into the next batch, exactly as the reader does.
//
// The compute kernel packs a block one subblock (subblock_M_tiles x subblock_N_tiles tiles, what DST
// holds) at a time, subblocks in row-major order over the block and tiles in row-major order within a
// subblock. This writer mirrors that order, maps every tile back to its (m_tile, n_tile) position in C and writes it
// by tile index through the tensor accessor. Tiles past M_tiles / N_tiles (edge blocks) are popped but
// not written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_batch = get_arg(args::first_batch);
    const uint32_t first_M_tile = get_arg(args::first_M_tile);
    const uint32_t first_N_tile = get_arg(args::first_N_tile);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr uint32_t per_core_M_tiles = get_arg(args::per_core_M_tiles);
    constexpr uint32_t per_core_N_tiles = get_arg(args::per_core_N_tiles);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);

    constexpr uint32_t C_tiles_per_batch = M_tiles * N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what the compute packs at once

    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    DataflowBuffer C_block(dfb::C_block);
    const uint32_t C_tile_bytes = C_block.get_entry_size();

    uint32_t batch = first_batch;
    uint32_t block_M_tile = first_M_tile;  // origin of the current block, in tiles
    uint32_t block_N_tile = first_N_tile;
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t C_batch_first_tile = batch * C_tiles_per_batch;

        // Same DST-group walk as the compute kernel: (m_tile, n_tile) is the group's first tile.
        for (uint32_t m_tile = 0; m_tile < per_core_M_tiles; m_tile += subblock_M_tiles) {
            for (uint32_t n_tile = 0; n_tile < per_core_N_tiles; n_tile += subblock_N_tiles) {
                C_block.wait_front(subblock_tiles);
                uint32_t slot_offset = 0;
                for (uint32_t tile_row = 0; tile_row < subblock_M_tiles; ++tile_row) {
                    const uint32_t C_m_tile = block_M_tile + m_tile + tile_row;  // tile position in C
                    for (uint32_t tile_column = 0; tile_column < subblock_N_tiles;
                         ++tile_column, slot_offset += C_tile_bytes) {
                        const uint32_t C_n_tile = block_N_tile + n_tile + tile_column;
                        if (C_m_tile < M_tiles && C_n_tile < N_tiles) {
                            noc.async_write(
                                C_block,
                                C,
                                C_tile_bytes,
                                {.offset_bytes = slot_offset},
                                {.page_id = C_batch_first_tile + C_m_tile * N_tiles + C_n_tile});
                        }
                    }
                }
                noc.async_write_barrier();
                C_block.pop_front(subblock_tiles);
            }
        }

        // Next block: across N, then down M, then the next batch.
        block_N_tile += per_core_N_tiles;
        if (block_N_tile >= N_tiles) {
            block_N_tile = 0;
            block_M_tile += per_core_M_tiles;
            if (block_M_tile >= M_tiles) {
                block_M_tile = 0;
                ++batch;
            }
        }
    }
}
