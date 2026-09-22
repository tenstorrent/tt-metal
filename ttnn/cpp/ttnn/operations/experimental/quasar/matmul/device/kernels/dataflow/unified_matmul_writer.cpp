// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer. The compute kernel packs a C slice one subblock at a time, subblocks
// row-major over the C slice and tiles row-major within a subblock; this writer mirrors that order,
// maps every tile back to its position in C and writes it by tile index through the tensor accessor.
// Tiles past M_tiles / N_tiles (edge C slices) are popped but not written.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_C_slices = get_arg(args::num_C_slices);

    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr uint32_t C_slice_M_tiles = get_arg(args::C_slice_M_tiles);
    constexpr uint32_t C_slice_N_tiles = get_arg(args::C_slice_N_tiles);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);
    constexpr bool C_borrowed = get_arg(args::C_borrowed) != 0;  // C's shard is the C_slice

    constexpr uint32_t C_batch_stride_tiles = M_tiles * N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what the compute packs at once

    DataflowBuffer C_slice(dfb::C_slice);
    if constexpr (C_borrowed) {
        // The C_slice IS this core's C shard: the compute packs the finished tiles in place, so there is
        // nothing to move. Wait for the whole C slice so the DFB's credits balance. (Borrowing needs one C slice
        // per core and batch 1, so this is the entire output.)
        C_slice.wait_front(C_slice_M_tiles * C_slice_N_tiles);
        return;
    }
    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    // One DFB entry per tile; a subblock sits in the DFB row-major in tiles, as the compute packs it.
    const uint32_t C_tile_bytes = get_tile_size(dfb::C_slice);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        const uint32_t C_batch_first_tile = batch * C_batch_stride_tiles;

        // Origin of the C slice being produced, in tiles. The host passes the origin of this cluster's first
        // C slice; the loop steps it across N, then down M, so every batch starts over from the argument.
        uint32_t C_slice_first_M_tile = get_arg(args::C_slice_first_M_tile);
        uint32_t C_slice_first_N_tile = get_arg(args::C_slice_first_N_tile);
        for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
            // Same subblock walk as the compute kernel: (m_tile, n_tile) is the subblock's first tile within
            // the C slice.
            for (uint32_t m_tile = 0; m_tile < C_slice_M_tiles; m_tile += subblock_M_tiles) {
                const uint32_t C_m_tile = C_slice_first_M_tile + m_tile;  // subblock's first row in C, in tiles
                for (uint32_t n_tile = 0; n_tile < C_slice_N_tiles; n_tile += subblock_N_tiles) {
                    const uint32_t C_n_tile = C_slice_first_N_tile + n_tile;  // subblock's first column in C
                    // Every subblock is waited for and popped, clipped or not, so the DFB's credits balance;
                    // only the tiles inside C are written.
                    C_slice.wait_front(subblock_tiles);
                    for (uint32_t subblock_m_tile = 0;
                         subblock_m_tile < subblock_M_tiles && C_m_tile + subblock_m_tile < M_tiles;
                         ++subblock_m_tile) {
                        const uint32_t C_row_first_tile =
                            C_batch_first_tile + (C_m_tile + subblock_m_tile) * N_tiles + C_n_tile;
                        const uint32_t row_offset_bytes = subblock_m_tile * subblock_N_tiles * C_tile_bytes;
                        for (uint32_t subblock_n_tile = 0;
                             subblock_n_tile < subblock_N_tiles && C_n_tile + subblock_n_tile < N_tiles;
                             ++subblock_n_tile) {
                            noc.async_write(
                                C_slice,
                                C,
                                C_tile_bytes,
                                {.offset_bytes = row_offset_bytes + subblock_n_tile * C_tile_bytes},
                                {.page_id = C_row_first_tile + subblock_n_tile});
                        }
                    }
                    noc.async_write_barrier();
                    C_slice.pop_front(subblock_tiles);
                }
            }

            // Next C slice: across N, then down M.
            C_slice_first_N_tile += C_slice_N_tiles;
            if (C_slice_first_N_tile >= N_tiles) {
                C_slice_first_N_tile = 0;
                C_slice_first_M_tile += C_slice_M_tiles;
            }
        }
    }
}
