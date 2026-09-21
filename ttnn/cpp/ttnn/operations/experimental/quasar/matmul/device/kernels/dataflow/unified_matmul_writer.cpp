// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer: stores this cluster's finished MN chunks.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N], batch_size times. An MN chunk is the MN_chunk_M_tiles x
// MN_chunk_N_tiles tiles of C at origin (MN_chunk_first_M_tile, MN_chunk_first_N_tile). This cluster owns num_MN_chunks
// consecutive chunks of the row-major walk over C (across N, then down M) starting at
// (first_MN_chunk_M_tile, first_MN_chunk_N_tile), and writes them for every batch, exactly as the reader
// walks them.
//
// The compute kernel packs a chunk one subblock (subblock_M_tiles x subblock_N_tiles tiles, what DST holds)
// at a time, subblocks in row-major order over the chunk and tiles in row-major order within a subblock.
// This writer mirrors that order, maps every tile back to its position in C and writes it by tile index
// through the tensor accessor. Tiles past M_tiles / N_tiles (edge chunks) are popped but not written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_MN_chunk_M_tile = get_arg(args::first_MN_chunk_M_tile);
    const uint32_t first_MN_chunk_N_tile = get_arg(args::first_MN_chunk_N_tile);
    const uint32_t num_MN_chunks = get_arg(args::num_MN_chunks);

    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr uint32_t MN_chunk_M_tiles = get_arg(args::MN_chunk_M_tiles);
    constexpr uint32_t MN_chunk_N_tiles = get_arg(args::MN_chunk_N_tiles);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);
    constexpr bool C_borrowed = get_arg(args::C_borrowed) != 0;  // C's shard is the C_slice ring

    constexpr uint32_t C_tiles_per_batch = M_tiles * N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what the compute packs at once

    DataflowBuffer C_slice(dfb::C_slice);
    if constexpr (C_borrowed) {
        // The C_slice ring IS this core's C shard: the compute packs the finished tiles in place, so there is
        // nothing to move. Wait for the whole chunk so the ring's credits balance. (Borrowing needs one chunk
        // per core and batch 1, so this is the entire output.)
        C_slice.wait_front(MN_chunk_M_tiles * MN_chunk_N_tiles);
        return;
    }
    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    const uint32_t C_tile_bytes = C_slice.get_entry_size();

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        const uint32_t C_batch_first_tile = batch * C_tiles_per_batch;

        uint32_t MN_chunk_first_M_tile = first_MN_chunk_M_tile;  // origin of the current chunk, in tiles
        uint32_t MN_chunk_first_N_tile = first_MN_chunk_N_tile;
        for (uint32_t MN_chunk = 0; MN_chunk < num_MN_chunks; ++MN_chunk) {
            // Same subblock walk as the compute kernel: (m_tile, n_tile) is the subblock's first tile within
            // the chunk.
            for (uint32_t m_tile = 0; m_tile < MN_chunk_M_tiles; m_tile += subblock_M_tiles) {
                for (uint32_t n_tile = 0; n_tile < MN_chunk_N_tiles; n_tile += subblock_N_tiles) {
                    C_slice.wait_front(subblock_tiles);
                    uint32_t slot_offset = 0;
                    for (uint32_t subblock_m_tile = 0; subblock_m_tile < subblock_M_tiles; ++subblock_m_tile) {
                        const uint32_t C_m_tile =
                            MN_chunk_first_M_tile + m_tile + subblock_m_tile;  // tile position in C
                        for (uint32_t subblock_n_tile = 0; subblock_n_tile < subblock_N_tiles;
                             ++subblock_n_tile, slot_offset += C_tile_bytes) {
                            const uint32_t C_n_tile = MN_chunk_first_N_tile + n_tile + subblock_n_tile;
                            if (C_m_tile < M_tiles && C_n_tile < N_tiles) {
                                noc.async_write(
                                    C_slice,
                                    C,
                                    C_tile_bytes,
                                    {.offset_bytes = slot_offset},
                                    {.page_id = C_batch_first_tile + C_m_tile * N_tiles + C_n_tile});
                            }
                        }
                    }
                    noc.async_write_barrier();
                    C_slice.pop_front(subblock_tiles);
                }
            }

            // Next chunk: across N, then down M.
            MN_chunk_first_N_tile += MN_chunk_N_tiles;
            if (MN_chunk_first_N_tile >= N_tiles) {
                MN_chunk_first_N_tile = 0;
                MN_chunk_first_M_tile += MN_chunk_M_tiles;
            }
        }
    }
}
