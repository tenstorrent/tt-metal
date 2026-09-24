// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul writer. The compute kernel packs a C slice one subblock at a time, subblocks
// row-major over the C slice and tiles row-major within a subblock; this writer mirrors that order,
// maps every tile back to its position in C and writes it by tile index through the tensor accessor.
// Tiles past the true C slice (subblock padding) or past M_tiles / N_tiles are popped but not
// written; padding overshoot may overlap a neighbouring core's C slice, so both clips are needed.
// Compile-time args are the template parameters, runtime args the function parameters.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t batch_size,
    uint32_t C_slice_M_tiles,
    uint32_t C_slice_N_tiles,
    uint32_t C_slice_M_padded_tiles,  // C slice dims rounded up to subblock multiples
    uint32_t C_slice_N_padded_tiles,
    uint32_t subblock_M_tiles,
    uint32_t subblock_N_tiles,
    uint32_t C_borrowed>  // C's L1 shard is the C_slice DFB: the compute packs in place, nothing is written
TT_KERNEL void writer(uint32_t first_C_slice, uint32_t num_C_slices) {
    // first_C_slice: this core's first C slice in the row-major walk over C (across N, then down M);
    // num_C_slices: how many consecutive ones it writes, per batch.
    constexpr uint32_t C_batch_stride_tiles = M_tiles * N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what the compute packs at once
    constexpr uint32_t C_slices_across_N = (N_tiles + C_slice_N_tiles - 1) / C_slice_N_tiles;

    DataflowBuffer C_slice(dfb::C_slice);
    if constexpr (C_borrowed) {
        // Consume the compute's credits for the whole shard so the DFB ends balanced.
        constexpr uint32_t C_shard_tiles = C_slice_M_padded_tiles * C_slice_N_padded_tiles;
        C_slice.wait_front(C_shard_tiles);
        C_slice.pop_front(C_shard_tiles);
        return;
    }
    const auto C = TensorAccessor(tensor::C);
    Noc noc;
    // One DFB entry per tile; a subblock sits in the DFB row-major in tiles, as the compute packs it.
    const uint32_t C_tile_bytes = get_tile_size(dfb::C_slice);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        const uint32_t C_batch_first_tile = batch * C_batch_stride_tiles;

        for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
            // Origin of this C slice, in tiles, from its position in the walk.
            const uint32_t C_slice_first_M_tile = ((first_C_slice + MN_chunk) / C_slices_across_N) * C_slice_M_tiles;
            const uint32_t C_slice_first_N_tile = ((first_C_slice + MN_chunk) % C_slices_across_N) * C_slice_N_tiles;
            // Same subblock walk as the compute kernel: (m_tile, n_tile) is the subblock's first tile within
            // the C slice.
            for (uint32_t m_tile = 0; m_tile < C_slice_M_padded_tiles; m_tile += subblock_M_tiles) {
                const uint32_t C_m_tile = C_slice_first_M_tile + m_tile;  // subblock's first row in C, in tiles
                for (uint32_t n_tile = 0; n_tile < C_slice_N_padded_tiles; n_tile += subblock_N_tiles) {
                    const uint32_t C_n_tile = C_slice_first_N_tile + n_tile;  // subblock's first column in C
                    // Every subblock is waited for and popped, clipped or not, so the DFB's credits balance;
                    // only the tiles inside C are written.
                    C_slice.wait_front(subblock_tiles);
                    for (uint32_t subblock_m_tile = 0;
                         subblock_m_tile < subblock_M_tiles && m_tile + subblock_m_tile < C_slice_M_tiles &&
                         C_m_tile + subblock_m_tile < M_tiles;
                         ++subblock_m_tile) {
                        const uint32_t C_row_first_tile =
                            C_batch_first_tile + (C_m_tile + subblock_m_tile) * N_tiles + C_n_tile;
                        const uint32_t row_offset_bytes = subblock_m_tile * subblock_N_tiles * C_tile_bytes;
                        for (uint32_t subblock_n_tile = 0;
                             subblock_n_tile < subblock_N_tiles && n_tile + subblock_n_tile < C_slice_N_tiles &&
                             C_n_tile + subblock_n_tile < N_tiles;
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
        }
    }
}
