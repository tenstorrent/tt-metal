// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer: stores this cluster's finished C subblocks.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N]. A C subblock is the per_core_M x per_core_N tiles of C
// at origin (subblock_M_tile, subblock_N_tile) in one batch. This cluster starts at (first_batch,
// first_M_tile, first_N_tile) and writes num_subblocks of them, stepping per_core_N tiles across N, then
// per_core_M tiles down M, then into the next batch, exactly as the reader does.
//
// The compute kernel packs a C subblock one DST group (dst_M_tiles x dst_N_tiles tiles) at a time, groups
// in row-major order over the subblock and tiles in row-major order within a group. This writer mirrors
// that order, maps every tile back to its (m, n) position in C and writes it by tile index through the
// tensor accessor. Tiles past M_tiles / N_tiles (edge subblocks) are popped but not written.

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
    const uint32_t num_subblocks = get_arg(args::num_subblocks);

    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t dst_M_tiles = get_arg(args::dst_M_tiles);
    constexpr uint32_t dst_N_tiles = get_arg(args::dst_N_tiles);

    constexpr uint32_t C_tiles_per_batch = M_tiles * N_tiles;
    constexpr uint32_t num_dst_tiles = dst_M_tiles * dst_N_tiles;  // C tiles the compute packs at once

    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    DataflowBuffer C_subblock(dfb::C_subblock);
    const uint32_t C_tile_bytes = C_subblock.get_entry_size();

    uint32_t batch = first_batch;
    uint32_t subblock_M_tile = first_M_tile;  // origin of the current C subblock, in tiles
    uint32_t subblock_N_tile = first_N_tile;
    for (uint32_t subblock = 0; subblock < num_subblocks; ++subblock) {
        const uint32_t C_batch_first_tile = batch * C_tiles_per_batch;

        // Same DST-group walk as the compute kernel: (dst_first_m, dst_first_n) is the group's first tile.
        for (uint32_t dst_first_m = 0; dst_first_m < per_core_M; dst_first_m += dst_M_tiles) {
            for (uint32_t dst_first_n = 0; dst_first_n < per_core_N; dst_first_n += dst_N_tiles) {
                C_subblock.wait_front(num_dst_tiles);
                uint32_t slot_offset = 0;
                for (uint32_t tile_row = 0; tile_row < dst_M_tiles; ++tile_row) {
                    const uint32_t m = subblock_M_tile + dst_first_m + tile_row;
                    for (uint32_t tile_column = 0; tile_column < dst_N_tiles;
                         ++tile_column, slot_offset += C_tile_bytes) {
                        const uint32_t n = subblock_N_tile + dst_first_n + tile_column;
                        if (m < M_tiles && n < N_tiles) {
                            noc.async_write(
                                C_subblock,
                                C,
                                C_tile_bytes,
                                {.offset_bytes = slot_offset},
                                {.page_id = C_batch_first_tile + m * N_tiles + n});
                        }
                    }
                }
                noc.async_write_barrier();
                C_subblock.pop_front(num_dst_tiles);
            }
        }

        // Next C subblock: across N, then down M, then the next batch.
        subblock_N_tile += per_core_N;
        if (subblock_N_tile >= N_tiles) {
            subblock_N_tile = 0;
            subblock_M_tile += per_core_M;
            if (subblock_M_tile >= M_tiles) {
                subblock_M_tile = 0;
                ++batch;
            }
        }
    }
}
