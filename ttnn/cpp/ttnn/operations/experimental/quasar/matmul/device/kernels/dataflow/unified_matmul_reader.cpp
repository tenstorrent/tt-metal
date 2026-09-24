// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul reader: per batch, C slice and K chunk it pushes one A slice
// ([C_slice_M_tiles][K_chunk_tiles] tiles) and one B slice ([K_chunk_tiles][C_slice_N_tiles]), both
// row-major, matching the compute kernel's loop order and indexing. Slices are sized to the
// subblock-padded C slice dims; tiles past the true slice or past M/N are never read (their stale
// entries only reach C tiles the writer drops). A's K-padding columns are zeroed.
// Compile-time args are the template parameters, runtime args the function parameters.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

template <
    uint32_t M_tiles,
    uint32_t K_tiles,
    uint32_t N_tiles,
    uint32_t batch_size,
    uint32_t B_batch_stride_tiles,  // 0 when B is a single [K x N] that every batch of A multiplies
    uint32_t C_slice_M_tiles,
    uint32_t C_slice_N_tiles,
    uint32_t C_slice_M_padded_tiles,  // C slice dims rounded up to subblock multiples
    uint32_t C_slice_N_padded_tiles,
    uint32_t K_chunk_tiles,
    uint32_t num_K_chunks,
    uint32_t A_last_K_tile_valid_columns,  // valid element columns in A's last K tile; 0 when K is a tile multiple
    uint32_t A_borrowed,                   // a borrowed operand is a resident L1 shard bound as the DFB: never read
    uint32_t B_borrowed>
TT_KERNEL void reader(uint32_t first_C_slice, uint32_t num_C_slices) {
    // first_C_slice: this core's first C slice in the row-major walk over C (across N, then down M);
    // num_C_slices: how many consecutive ones it produces, per batch.
    constexpr DataFormat A_format = get_dataformat(dfb::A_slice);
    constexpr uint32_t A_slice_tiles = C_slice_M_padded_tiles * K_chunk_tiles;
    constexpr uint32_t B_slice_tiles = K_chunk_tiles * C_slice_N_padded_tiles;
    constexpr uint32_t A_batch_stride_tiles = M_tiles * K_tiles;
    constexpr uint32_t C_slices_across_N = (N_tiles + C_slice_N_tiles - 1) / C_slice_N_tiles;

    Noc noc;
    DataflowBuffer A_slice(dfb::A_slice);
    DataflowBuffer B_slice(dfb::B_slice);

    [[maybe_unused]] const auto A = TensorAccessor(tensor::A);
    [[maybe_unused]] const auto B = TensorAccessor(tensor::B);

    // A borrowed operand's DFB IS its resident L1 shard: hand the whole DFB to the compute once and never
    // read it. (A: the single K chunk covers all of K; B: the compute consumes it one K chunk at a time.)
    if constexpr (A_borrowed) {
        A_slice.reserve_back(A_slice_tiles);
        A_slice.push_back(A_slice_tiles);
    }
    if constexpr (B_borrowed) {
        B_slice.reserve_back(K_tiles * C_slice_N_tiles);
        B_slice.push_back(K_tiles * C_slice_N_tiles);
    }

    // One DFB entry per tile: a slice is stored row-major in tiles, entry (row, column) at
    // (row * columns + column) * tile bytes, which is how the compute kernel indexes it.
    const uint32_t A_tile_bytes = get_tile_size(dfb::A_slice);
    const uint32_t B_tile_bytes = get_tile_size(dfb::B_slice);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        const uint32_t A_batch_first_tile = batch * A_batch_stride_tiles;
        const uint32_t B_batch_first_tile = batch * B_batch_stride_tiles;

        for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
            // Origin of this C slice, in tiles, from its position in the walk.
            const uint32_t C_slice_first_M_tile = ((first_C_slice + MN_chunk) / C_slices_across_N) * C_slice_M_tiles;
            const uint32_t C_slice_first_N_tile = ((first_C_slice + MN_chunk) % C_slices_across_N) * C_slice_N_tiles;
            for (uint32_t K_chunk = 0; K_chunk < num_K_chunks; ++K_chunk) {
                const uint32_t K_chunk_first_K_tile = K_chunk * K_chunk_tiles;

                if constexpr (!A_borrowed) {
                    // A slice: rows C_slice_first_M_tile.., columns K_chunk_first_K_tile.., entry (m_tile, k_tile).
                    // Rows past the edge of A are clipped: they trail, so they are simply not written.
                    A_slice.reserve_back(A_slice_tiles);
                    for (uint32_t m_tile = 0; m_tile < C_slice_M_tiles && C_slice_first_M_tile + m_tile < M_tiles;
                         ++m_tile) {
                        const uint32_t A_row_first_tile =
                            A_batch_first_tile + (C_slice_first_M_tile + m_tile) * K_tiles + K_chunk_first_K_tile;
                        const uint32_t A_row_offset_bytes = m_tile * K_chunk_tiles * A_tile_bytes;
                        for (uint32_t k_tile = 0; k_tile < K_chunk_tiles; ++k_tile) {
                            noc.async_read(
                                A,
                                A_slice,
                                A_tile_bytes,
                                {.page_id = A_row_first_tile + k_tile},
                                {.offset_bytes = A_row_offset_bytes + k_tile * A_tile_bytes});
                        }
                        if constexpr (A_last_K_tile_valid_columns > 0) {
                            // K is not a tile multiple, and this row's last tile is A's last K tile: once it has
                            // landed, zero its padding columns so they add nothing to C.
                            if (K_chunk == num_K_chunks - 1) {
                                noc.async_read_barrier();
                                pad_last_ktile<A_format, A_last_K_tile_valid_columns>(
                                    A_slice.get_write_ptr() + A_row_offset_bytes + (K_chunk_tiles - 1) * A_tile_bytes);
                            }
                        }
                    }
                }
                if constexpr (!B_borrowed) {
                    // B slice: rows K_chunk_first_K_tile.., columns C_slice_first_N_tile.., entry (k_tile, n_tile).
                    // Columns past the edge of B are clipped; they keep their entry, which only feeds clipped C.
                    B_slice.reserve_back(B_slice_tiles);
                    for (uint32_t k_tile = 0; k_tile < K_chunk_tiles; ++k_tile) {
                        const uint32_t B_row_first_tile =
                            B_batch_first_tile + (K_chunk_first_K_tile + k_tile) * N_tiles + C_slice_first_N_tile;
                        const uint32_t B_row_offset_bytes = k_tile * C_slice_N_padded_tiles * B_tile_bytes;
                        for (uint32_t n_tile = 0; n_tile < C_slice_N_tiles && C_slice_first_N_tile + n_tile < N_tiles;
                             ++n_tile) {
                            noc.async_read(
                                B,
                                B_slice,
                                B_tile_bytes,
                                {.page_id = B_row_first_tile + n_tile},
                                {.offset_bytes = B_row_offset_bytes + n_tile * B_tile_bytes});
                        }
                    }
                }
                noc.async_read_barrier();

                // A borrowed operand was published once above; only copied slices are pushed here.
                if constexpr (!A_borrowed) {
                    A_slice.push_back(A_slice_tiles);
                }
                if constexpr (!B_borrowed) {
                    B_slice.push_back(B_slice_tiles);
                }
            }
        }
    }
}
