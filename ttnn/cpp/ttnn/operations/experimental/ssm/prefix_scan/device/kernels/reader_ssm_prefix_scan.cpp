// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "experimental/noc.h"

constexpr uint32_t NUM_TILES_IN_TILIZED_CHUNK = 32;

void kernel_main() {
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bx_in_id = get_compile_time_arg_val(1);
    static_assert(cb_bx_in_id == cb_in_id, "cb_bx_in_id must match cb_in_id because bx reuses the shared staging CB");
    constexpr uint32_t cb_h_in_id = get_compile_time_arg_val(2);
    constexpr uint32_t input_tile_size = get_compile_time_arg_val(3);
    constexpr uint32_t h_page_size = get_compile_time_arg_val(4);

    const uint32_t total_tiles_per_row = get_arg_val<uint32_t>(0);
    const uint32_t total_tiles_per_col = get_arg_val<uint32_t>(1);
    const uint32_t a_shard_l1_addr = get_arg_val<uint32_t>(2);
    const uint32_t bx_shard_l1_addr = get_arg_val<uint32_t>(3);
    const uint32_t h_shard_l1_addr = get_arg_val<uint32_t>(4);

    experimental::CircularBuffer cb_in(cb_in_id);
    experimental::CircularBuffer cb_h_in(cb_h_in_id);
    experimental::Noc noc;
    experimental::UnicastEndpoint shard_src;

    const uint32_t local_noc_x = my_x[noc.get_noc_id()];
    const uint32_t local_noc_y = my_y[noc.get_noc_id()];

    const uint32_t num_chunks_per_row =
        (total_tiles_per_row + NUM_TILES_IN_TILIZED_CHUNK - 1) / NUM_TILES_IN_TILIZED_CHUNK;

    const uint32_t chunk_bytes = NUM_TILES_IN_TILIZED_CHUNK * input_tile_size;

    // Copy h_prev data from shard to staging CB via NOC read.
    // cb_h_in is non-shard-backed to keep it at a low L1 address, avoiding unpacker OOB
    // when the h_prev shard is near the L1 boundary.
    const uint32_t h_total_bytes = total_tiles_per_row * h_page_size;
    cb_h_in.reserve_back(total_tiles_per_row);
    noc.async_read(
        shard_src,
        cb_h_in,
        h_total_bytes,
        {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = h_shard_l1_addr},
        {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_h_in.push_back(total_tiles_per_row);

    // Copy tiles from shard to staging CB chunk by chunk via NOC read.
    // A single staging CB is shared between a and bx: the reader pushes a's chunk, compute
    // consumes it (untilize), then the reader pushes bx's chunk and compute consumes that.
    // Each staging chunk always contains 32 tiles. For row tails shorter than 32 tiles,
    // read only the remaining tiles from the current row and zero-pad the rest so the
    // consumer never observes data from the next row.
    for (uint32_t row = 0; row < total_tiles_per_col; row++) {
        for (uint32_t chunk = 0; chunk < num_chunks_per_row; chunk++) {
            const uint32_t tile_offset_in_row = chunk * NUM_TILES_IN_TILIZED_CHUNK;
            const uint32_t tile_offset = row * total_tiles_per_row + tile_offset_in_row;
            const uint32_t byte_offset = tile_offset * input_tile_size;
            const uint32_t tiles_remaining_in_row =
                tile_offset_in_row < total_tiles_per_row ? (total_tiles_per_row - tile_offset_in_row) : 0;
            const uint32_t tiles_to_read = tiles_remaining_in_row >= NUM_TILES_IN_TILIZED_CHUNK
                                               ? NUM_TILES_IN_TILIZED_CHUNK
                                               : tiles_remaining_in_row;
            const uint32_t bytes_to_copy = tiles_to_read * input_tile_size;

            // Copy a chunk to shared staging CB
            cb_in.reserve_back(NUM_TILES_IN_TILIZED_CHUNK);
            noc.async_read(
                shard_src,
                cb_in,
                bytes_to_copy,
                {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = a_shard_l1_addr + byte_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            if (bytes_to_copy < chunk_bytes) {
                experimental::CoreLocalMem<volatile uint32_t> pad(cb_in.get_write_ptr() + bytes_to_copy);
                const uint32_t padding_words = (chunk_bytes - bytes_to_copy) / sizeof(uint32_t);
                for (uint32_t w = 0; w < padding_words; w++) {
                    pad[w] = 0;
                }
            }
            cb_in.push_back(NUM_TILES_IN_TILIZED_CHUNK);

            // Copy bx chunk to the same shared staging CB (compute pops a first, freeing space)
            cb_in.reserve_back(NUM_TILES_IN_TILIZED_CHUNK);
            noc.async_read(
                shard_src,
                cb_in,
                bytes_to_copy,
                {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = bx_shard_l1_addr + byte_offset},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            if (bytes_to_copy < chunk_bytes) {
                experimental::CoreLocalMem<volatile uint32_t> pad(cb_in.get_write_ptr() + bytes_to_copy);
                const uint32_t padding_words = (chunk_bytes - bytes_to_copy) / sizeof(uint32_t);
                for (uint32_t w = 0; w < padding_words; w++) {
                    pad[w] = 0;
                }
            }
            cb_in.push_back(NUM_TILES_IN_TILIZED_CHUNK);
        }
    }
}
