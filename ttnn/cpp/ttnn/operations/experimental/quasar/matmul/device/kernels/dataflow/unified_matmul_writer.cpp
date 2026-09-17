// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer: stores this cluster's finished C blocks.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N]. Work item w is C block w % num_C_blocks of batch
// w / num_C_blocks; this cluster owns a contiguous run of items. The compute kernel packs each C block one DST subblock
// (subblock_M_tiles x subblock_N_tiles) at a time, subblocks in row-major order over the block and tiles
// in row-major order within a subblock. This writer mirrors that order, maps every tile back to its
// (m, n) position in C and writes it by tile index through the tensor accessor. Tiles past M_tiles /
// N_tiles (edge blocks) are popped but not written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_work_item = get_arg(args::first_work_item);
    const uint32_t num_work_items = get_arg(args::num_work_items);

    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr uint32_t num_C_blocks = get_arg(args::num_C_blocks);  // per batch
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);
    constexpr uint32_t num_C_block_columns = get_arg(args::num_C_block_columns);

    constexpr uint32_t C_tiles_per_batch = M_tiles * N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;
    constexpr uint32_t num_subblock_rows = per_core_M / subblock_M_tiles;
    constexpr uint32_t num_subblock_columns = per_core_N / subblock_N_tiles;

    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    DataflowBuffer C_block(dfb::C_block);
    const uint32_t C_tile_bytes = C_block.get_entry_size();

    for (uint32_t work_item = first_work_item; work_item < first_work_item + num_work_items; ++work_item) {
        const uint32_t batch = work_item / num_C_blocks;
        const uint32_t C_block_index = work_item % num_C_blocks;
        const uint32_t C_batch_first_tile = batch * C_tiles_per_batch;
        const uint32_t first_M_tile = (C_block_index / num_C_block_columns) * per_core_M;
        const uint32_t first_N_tile = (C_block_index % num_C_block_columns) * per_core_N;
        {
            for (uint32_t subblock_row = 0; subblock_row < num_subblock_rows; ++subblock_row) {
                for (uint32_t subblock_column = 0; subblock_column < num_subblock_columns; ++subblock_column) {
                    C_block.wait_front(subblock_tiles);
                    uint32_t slot_offset = 0;
                    for (uint32_t tile_row = 0; tile_row < subblock_M_tiles; ++tile_row) {
                        const uint32_t m = first_M_tile + subblock_row * subblock_M_tiles + tile_row;
                        for (uint32_t tile_column = 0; tile_column < subblock_N_tiles;
                             ++tile_column, slot_offset += C_tile_bytes) {
                            const uint32_t n = first_N_tile + subblock_column * subblock_N_tiles + tile_column;
                            if (m < M_tiles && n < N_tiles) {
                                noc.async_write(
                                    C_block,
                                    C,
                                    C_tile_bytes,
                                    {.offset_bytes = slot_offset},
                                    {.page_id = C_batch_first_tile + m * N_tiles + n});
                            }
                        }
                    }
                    noc.async_write_barrier();
                    C_block.pop_front(subblock_tiles);
                }
            }
        }
    }
}
