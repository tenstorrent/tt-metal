// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul reader: streams the A and B slices for this cluster's C chunks of C.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N] = A[M x K] x B[K x N]. A C chunk is the per_core_M_tiles x
// per_core_N_tiles tiles of C at origin (C_chunk_M_tile, C_chunk_N_tile) in one batch. This cluster starts at
// (first_batch, first_C_chunk_M_tile, first_C_chunk_N_tile) and produces num_C_chunks of them, stepping
// per_core_N_tiles tiles across N, then per_core_M_tiles tiles down M, then into the next batch. For each C chunk and
// each K iteration the reader pushes
//   - one A slice: the C chunk's rows of A, K_iteration_tiles wide    -> [per_core_M_tiles][K_iteration_tiles] tiles
//   - one B slice: the C chunk's columns of B, K_iteration_tiles tall -> [K_iteration_tiles][per_core_N_tiles] tiles
// both row-major, which is the layout the compute kernel indexes. Loop order (C chunk, K iteration) matches it.
//
// Edge C chunks: tiles past M_tiles / N_tiles are never read; their slots keep stale L1, which only reaches
// C tiles the writer drops (a valid C tile uses valid A rows and valid B columns only). When K is not a
// multiple of the tile dim, the padding columns of A's last K tile are zeroed so they contribute nothing.
//
// A and B are addressed by tile index through the tensor accessor (page id == row-major tile index within
// the tensor), so interleaved, L1-sharded and DRAM-sharded inputs are one code path.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    // Per-core runtime args: where this cluster's run of C chunks starts and how long it is.
    const uint32_t first_batch = get_arg(args::first_batch);
    const uint32_t first_C_chunk_M_tile = get_arg(args::first_C_chunk_M_tile);
    const uint32_t first_C_chunk_N_tile = get_arg(args::first_C_chunk_N_tile);
    const uint32_t num_C_chunks = get_arg(args::num_C_chunks);

    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t K_tiles = get_arg(args::K_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);
    constexpr bool broadcast_B_over_batch = get_arg(args::broadcast_B_over_batch) != 0;
    constexpr uint32_t per_core_M_tiles = get_arg(args::per_core_M_tiles);
    constexpr uint32_t per_core_N_tiles = get_arg(args::per_core_N_tiles);
    constexpr uint32_t K_iteration_tiles = get_arg(args::K_iteration_tiles);
    constexpr uint32_t num_K_iterations = get_arg(args::num_K_iterations);
    // Valid element columns in A's last K tile; 0 when K is a multiple of the tile dim.
    constexpr uint32_t A_last_K_tile_valid_columns = get_arg(args::A_last_K_tile_valid_columns);

    constexpr uint32_t A_slice_tiles = per_core_M_tiles * K_iteration_tiles;
    constexpr uint32_t B_slice_tiles = K_iteration_tiles * per_core_N_tiles;
    constexpr uint32_t A_tiles_per_batch = M_tiles * K_tiles;
    constexpr uint32_t B_tiles_per_batch = K_tiles * N_tiles;

    const auto A = TensorAccessor(tensor::A);
    const auto B = TensorAccessor(tensor::B);
    Noc noc;
    DataflowBuffer A_slice(dfb::A_slice);
    DataflowBuffer B_slice(dfb::B_slice);

    const uint32_t A_tile_bytes = get_tile_size(dfb::A_slice);
    const uint32_t B_tile_bytes = get_tile_size(dfb::B_slice);
    // A ring slot is one tile at the DRAM-aligned stride the factory sized the ring with (== tile bytes
    // for every 32x32 format).
    const uint32_t A_slot_bytes = A_slice.get_entry_size();
    const uint32_t B_slot_bytes = B_slice.get_entry_size();

    uint32_t batch = first_batch;
    uint32_t C_chunk_M_tile = first_C_chunk_M_tile;  // origin of the current C chunk, in tiles
    uint32_t C_chunk_N_tile = first_C_chunk_N_tile;
    for (uint32_t C_chunk_index = 0; C_chunk_index < num_C_chunks; ++C_chunk_index) {
        const uint32_t A_batch_first_tile = batch * A_tiles_per_batch;
        const uint32_t B_batch_first_tile = broadcast_B_over_batch ? 0 : batch * B_tiles_per_batch;
        // Rows / columns of this C chunk that lie inside the matrices (edge C chunks are clipped).
        const uint32_t valid_M_tiles =
            (M_tiles - C_chunk_M_tile < per_core_M_tiles) ? (M_tiles - C_chunk_M_tile) : per_core_M_tiles;
        const uint32_t valid_N_tiles =
            (N_tiles - C_chunk_N_tile < per_core_N_tiles) ? (N_tiles - C_chunk_N_tile) : per_core_N_tiles;

        for (uint32_t K_iteration = 0; K_iteration < num_K_iterations; ++K_iteration) {
            const uint32_t first_K_tile = K_iteration * K_iteration_tiles;

            // A slice: rows C_chunk_M_tile.., columns first_K_tile.. (invalid rows trail, so they are
            // simply not written).
            A_slice.reserve_back(A_slice_tiles);
            uint32_t A_slot_offset = 0;
            for (uint32_t m_tile = 0; m_tile < valid_M_tiles; ++m_tile) {
                uint32_t A_tile_index = A_batch_first_tile + (C_chunk_M_tile + m_tile) * K_tiles + first_K_tile;
                for (uint32_t k_tile = 0; k_tile < K_iteration_tiles;
                     ++k_tile, ++A_tile_index, A_slot_offset += A_slot_bytes) {
                    noc.async_read(
                        A, A_slice, A_tile_bytes, {.page_id = A_tile_index}, {.offset_bytes = A_slot_offset});
                }
            }

            // B slice: rows first_K_tile.., columns C_chunk_N_tile.. (invalid columns keep their slot).
            B_slice.reserve_back(B_slice_tiles);
            uint32_t B_slot_offset = 0;
            for (uint32_t k_tile = 0; k_tile < K_iteration_tiles; ++k_tile) {
                uint32_t B_tile_index = B_batch_first_tile + (first_K_tile + k_tile) * N_tiles + C_chunk_N_tile;
                for (uint32_t n_tile = 0; n_tile < per_core_N_tiles;
                     ++n_tile, ++B_tile_index, B_slot_offset += B_slot_bytes) {
                    if (n_tile < valid_N_tiles) {
                        noc.async_read(
                            B, B_slice, B_tile_bytes, {.page_id = B_tile_index}, {.offset_bytes = B_slot_offset});
                    }
                }
            }

            noc.async_read_barrier();

            if constexpr (A_last_K_tile_valid_columns > 0) {
                // Zero the padding columns of the last K tile in every valid row (reads have landed).
                if (K_iteration == num_K_iterations - 1) {
                    constexpr DataFormat A_format = get_dataformat(dfb::A_slice);
                    const uint32_t last_K_tile_of_row_0 =
                        A_slice.get_write_ptr() + (K_iteration_tiles - 1) * A_slot_bytes;
                    for (uint32_t m_tile = 0; m_tile < valid_M_tiles; ++m_tile) {
                        pad_last_ktile<A_format, A_last_K_tile_valid_columns>(
                            last_K_tile_of_row_0 + m_tile * K_iteration_tiles * A_slot_bytes);
                    }
                }
            }

            A_slice.push_back(A_slice_tiles);
            B_slice.push_back(B_slice_tiles);
        }

        // Next C chunk: across N, then down M, then the next batch.
        C_chunk_N_tile += per_core_N_tiles;
        if (C_chunk_N_tile >= N_tiles) {
            C_chunk_N_tile = 0;
            C_chunk_M_tile += per_core_M_tiles;
            if (C_chunk_M_tile >= M_tiles) {
                C_chunk_M_tile = 0;
                ++batch;
            }
        }
    }
}
