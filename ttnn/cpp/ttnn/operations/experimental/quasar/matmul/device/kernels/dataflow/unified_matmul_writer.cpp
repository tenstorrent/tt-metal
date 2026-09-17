// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer: stores this cluster's finished C subblocks.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N]. Work item w is C subblock w % C_subblocks_per_batch of
// batch w / C_subblocks_per_batch; this cluster owns a contiguous run of items. The compute kernel packs a
// C subblock one DST group (dst_M_tiles x dst_N_tiles tiles) at a time, groups in row-major order over the
// subblock and tiles in row-major order within a group. This writer mirrors that order, maps every tile
// back to its (m, n) position in C and writes it by tile index through the tensor accessor. Tiles past
// M_tiles / N_tiles (edge subblocks) are popped but not written.

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
    constexpr uint32_t C_subblocks_per_batch = get_arg(args::C_subblocks_per_batch);  // per batch
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t dst_M_tiles = get_arg(args::dst_M_tiles);
    constexpr uint32_t dst_N_tiles = get_arg(args::dst_N_tiles);
    constexpr uint32_t C_subblock_grid_columns = get_arg(args::C_subblock_grid_columns);

    constexpr uint32_t C_tiles_per_batch = M_tiles * N_tiles;
    constexpr uint32_t num_dst_tiles = dst_M_tiles * dst_N_tiles;  // C tiles the compute packs at once

    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    DataflowBuffer C_subblock(dfb::C_subblock);
    const uint32_t C_tile_bytes = C_subblock.get_entry_size();

    for (uint32_t work_item = first_work_item; work_item < first_work_item + num_work_items; ++work_item) {
        const uint32_t batch = work_item / C_subblocks_per_batch;
        const uint32_t C_subblock_index = work_item % C_subblocks_per_batch;
        const uint32_t C_batch_first_tile = batch * C_tiles_per_batch;
        const uint32_t first_M_tile = (C_subblock_index / C_subblock_grid_columns) * per_core_M;
        const uint32_t first_N_tile = (C_subblock_index % C_subblock_grid_columns) * per_core_N;
        {
            // Same DST-group walk as the compute kernel: (dst_first_m, dst_first_n) is the group's first tile.
            for (uint32_t dst_first_m = 0; dst_first_m < per_core_M; dst_first_m += dst_M_tiles) {
                for (uint32_t dst_first_n = 0; dst_first_n < per_core_N; dst_first_n += dst_N_tiles) {
                    C_subblock.wait_front(num_dst_tiles);
                    uint32_t slot_offset = 0;
                    for (uint32_t tile_row = 0; tile_row < dst_M_tiles; ++tile_row) {
                        const uint32_t m = first_M_tile + dst_first_m + tile_row;
                        for (uint32_t tile_column = 0; tile_column < dst_N_tiles;
                             ++tile_column, slot_offset += C_tile_bytes) {
                            const uint32_t n = first_N_tile + dst_first_n + tile_column;
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
        }
    }
}
