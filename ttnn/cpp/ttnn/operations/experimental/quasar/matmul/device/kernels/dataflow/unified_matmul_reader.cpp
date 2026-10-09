// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul reader: per batch, C slice and K chunk it pushes one A slice
// ([C_slice_M_tiles][K_chunk_tiles] tiles) and one B slice ([K_chunk_tiles][C_slice_N_tiles]), both
// row-major, matching the compute kernel's loop order and indexing. Slices are sized to the
// subblock-padded C slice dims; tiles past the true slice or past M/N are never read (their stale
// entries only reach C tiles the writer drops). A's K-padding columns are zeroed.
// Reader thread t (Quasar DM cores; one thread elsewhere) reads K chunks t, t + num_reader_threads, ... of every
// C slice into its own part of the A and B DFBs, and the compute's waits take the threads' parts in turn.
// K_chunks_per_C_slice_padded rounds the K chunks up to the reader threads; the extra ones carry credits only.
// With multicast, the first core of a row reads the row's A slices and multicasts them to the rest of the row,
// and the first core of a column does the same with B; a receiving core reads none of that operand.
// Compile-time args are the template parameters, runtime args the function parameters.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/semaphore.h"
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
    uint32_t K_chunks_per_C_slice,
    uint32_t K_chunks_per_C_slice_padded,
    uint32_t A_last_K_tile_valid_columns,  // valid element columns in A's last K tile; 0 when K is a tile multiple
    uint32_t A_borrowed,                   // a borrowed operand is a resident L1 shard bound as the DFB: never read
    uint32_t B_borrowed,
    uint32_t num_reader_threads,  // more than one only with copied A and B
    uint32_t A_mcast_num_dests,   // cores an A sender multicasts to (its row minus itself); 0 = no A multicast
    uint32_t B_mcast_num_dests>   // cores a B sender multicasts to (its column minus itself); 0 = no B multicast
TT_KERNEL void reader(
    uint32_t first_C_slice,
    uint32_t num_C_slices,
    uint32_t A_mcast_receiver,  // 1 when this core's A slices come from the first core of its row
    uint32_t A_mcast_sender_x,  // NoC coordinates of that core, which multicasts along the row to A_mcast_end_x
    uint32_t A_mcast_sender_y,
    uint32_t A_mcast_end_x,
    uint32_t B_mcast_receiver,  // the same for B and the first core of this core's column, down to B_mcast_end_y
    uint32_t B_mcast_sender_x,
    uint32_t B_mcast_sender_y,
    uint32_t B_mcast_end_y) {
    // first_C_slice: this core's first C slice in the row-major walk over C (across N, then down M);
    // num_C_slices: how many consecutive ones it produces, per batch.
    constexpr DataFormat A_format = get_dataformat(dfb::A_slice);
    constexpr uint32_t A_slice_tiles = C_slice_M_padded_tiles * K_chunk_tiles;
    constexpr uint32_t B_slice_tiles = K_chunk_tiles * C_slice_N_padded_tiles;
    constexpr uint32_t A_batch_stride_tiles = M_tiles * K_tiles;
    constexpr uint32_t C_slices_across_N = (N_tiles + C_slice_N_tiles - 1) / C_slice_N_tiles;
    const uint32_t thread = get_my_thread_id();

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

    // Multicast handshake per slice: a receiver clears its data_ready flag and, once its DFB has room, counts
    // itself into the sender's receivers_ready; the sender then multicasts the slice and its own VALID flag.
    Semaphore A_receivers_ready(sem::A_receivers_ready);
    Semaphore A_data_ready(sem::A_data_ready);
    Semaphore B_receivers_ready(sem::B_receivers_ready);
    Semaphore B_data_ready(sem::B_data_ready);
    A_data_ready.set(VALID);
    B_data_ready.set(VALID);
    const uint32_t A_rows_to_read = A_mcast_receiver ? 0 : C_slice_M_tiles;
    const uint32_t B_rows_to_read = B_mcast_receiver ? 0 : K_chunk_tiles;

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        const uint32_t A_batch_first_tile = batch * A_batch_stride_tiles;
        const uint32_t B_batch_first_tile = batch * B_batch_stride_tiles;

        for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
            // Origin of this C slice, in tiles, from its position in the walk.
            const uint32_t C_slice_first_M_tile = ((first_C_slice + MN_chunk) / C_slices_across_N) * C_slice_M_tiles;
            const uint32_t C_slice_first_N_tile = ((first_C_slice + MN_chunk) % C_slices_across_N) * C_slice_N_tiles;
            for (uint32_t K_chunk = thread; K_chunk < K_chunks_per_C_slice_padded; K_chunk += num_reader_threads) {
                if (K_chunk >= K_chunks_per_C_slice) {
                    A_slice.reserve_back(A_slice_tiles);
                    A_slice.push_back(A_slice_tiles);
                    B_slice.reserve_back(B_slice_tiles);
                    B_slice.push_back(B_slice_tiles);
                    continue;
                }
                const uint32_t K_chunk_first_K_tile = K_chunk * K_chunk_tiles;

                if constexpr (!A_borrowed) {
                    // A slice: rows C_slice_first_M_tile.., columns K_chunk_first_K_tile.., entry (m_tile, k_tile).
                    // Rows past the edge of A are clipped: they trail, so they are simply not written.
                    A_slice.reserve_back(A_slice_tiles);
                    if (A_mcast_receiver) {
                        A_data_ready.set(INVALID);
                        A_receivers_ready.up(noc, A_mcast_sender_x, A_mcast_sender_y, 1);
                    }
                    for (uint32_t m_tile = 0; m_tile < A_rows_to_read && C_slice_first_M_tile + m_tile < M_tiles;
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
                            if (K_chunk == K_chunks_per_C_slice - 1) {
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
                    if (B_mcast_receiver) {
                        B_data_ready.set(INVALID);
                        B_receivers_ready.up(noc, B_mcast_sender_x, B_mcast_sender_y, 1);
                    }
                    for (uint32_t k_tile = 0; k_tile < B_rows_to_read; ++k_tile) {
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

                if constexpr (A_mcast_num_dests > 0) {
                    if (A_mcast_receiver) {
                        A_data_ready.wait(VALID);
                    } else {
                        A_receivers_ready.wait(A_mcast_num_dests);
                        A_receivers_ready.set(0);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(A_slice.get_write_ptr()),
                            MulticastEndpoint{},
                            A_slice_tiles * A_tile_bytes,
                            A_mcast_num_dests,
                            {},
                            {.noc_x_start = A_mcast_sender_x,
                             .noc_y_start = A_mcast_sender_y,
                             .noc_x_end = A_mcast_end_x,
                             .noc_y_end = A_mcast_sender_y,
                             .addr = A_slice.get_write_ptr()},
                            /*linked=*/true);
                        A_data_ready.set_multicast(
                            noc,
                            A_mcast_sender_x,
                            A_mcast_sender_y,
                            A_mcast_end_x,
                            A_mcast_sender_y,
                            A_mcast_num_dests);
                    }
                }
                if constexpr (B_mcast_num_dests > 0) {
                    if (B_mcast_receiver) {
                        B_data_ready.wait(VALID);
                    } else {
                        B_receivers_ready.wait(B_mcast_num_dests);
                        B_receivers_ready.set(0);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(B_slice.get_write_ptr()),
                            MulticastEndpoint{},
                            B_slice_tiles * B_tile_bytes,
                            B_mcast_num_dests,
                            {},
                            {.noc_x_start = B_mcast_sender_x,
                             .noc_y_start = B_mcast_sender_y,
                             .noc_x_end = B_mcast_sender_x,
                             .noc_y_end = B_mcast_end_y,
                             .addr = B_slice.get_write_ptr()},
                            /*linked=*/true);
                        B_data_ready.set_multicast(
                            noc,
                            B_mcast_sender_x,
                            B_mcast_sender_y,
                            B_mcast_sender_x,
                            B_mcast_end_y,
                            B_mcast_num_dests);
                    }
                }

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
    if constexpr (A_mcast_num_dests > 0 || B_mcast_num_dests > 0) {
        noc.async_write_barrier();
        noc.async_atomic_barrier();
    }
}
