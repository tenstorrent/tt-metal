// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "experimental/kernel_args.h"
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_ring_id = get_arg(args::my_ring_id);
constexpr uint32_t ring_size = get_arg(args::ring_size);
constexpr uint32_t num_pages_per_packet = get_arg(args::packet_size_in_pages);
constexpr uint32_t chunk_granularity = get_arg(args::chunk_granularity);
constexpr uint32_t chunk_num_tiles = get_arg(args::chunk_num_tiles);
constexpr uint32_t num_chunks_per_shard = get_arg(args::num_chunks_per_shard);
constexpr uint32_t page_size = get_arg(args::page_size);
constexpr uint32_t contig_pages_advanced = get_arg(args::contig_pages_advanced);
constexpr uint32_t N_DRAM_BANKS = get_arg(args::num_dram_banks);

constexpr uint32_t NUM_SENDERS = ring_size - 1;

void kernel_main() {
    experimental::GlobalSemaphore global_semaphore(global_sem::semaphore);
    uint32_t in_row_tiles = get_arg(args::in_row_tiles);
    uint32_t in_col_tiles = get_arg(args::in_col_tiles);
    uint32_t input_row_device_stride = get_arg(args::input_row_device_stride);
    uint32_t input_col_device_stride = get_arg(args::input_col_device_stride);
    uint32_t input_shard_row_tiles = get_arg(args::input_shard_row_tiles);
    uint32_t input_shard_col_tiles = get_arg(args::input_shard_col_tiles);
    uint32_t out_row_start = get_arg(args::out_row_start);
    uint32_t out_col_start = get_arg(args::out_col_start);
    uint32_t out_row_tiles = get_arg(args::out_row_tiles);
    uint32_t out_col_tiles = get_arg(args::out_col_tiles);
    uint32_t num_pages_per_packet = get_arg(args::num_pages_per_packet);
    uint32_t remote_device_ring_id = get_arg(args::remote_device_ring_id);

    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    auto input_tensor_addrgen = TensorAccessor(tensor::input);
    auto intermediate_tensor_addrgen = TensorAccessor(tensor::intermediate);

    Noc noc_obj;
    DataflowBuffer dfb(dfb::receiver_data);

    if (my_ring_id == remote_device_ring_id) {
        // Follows same logic as sender reader for local copy.
        uint32_t shard_row_start_id = my_ring_id * input_row_device_stride;
        uint32_t shard_col_start_id = my_ring_id * input_col_device_stride;
        uint32_t shard_row_end_id = shard_row_start_id + input_shard_row_tiles;
        uint32_t shard_col_end_id = shard_col_start_id + input_shard_col_tiles;

        for (uint32_t row_tile_id = shard_row_start_id; row_tile_id < shard_row_end_id; row_tile_id++) {
            for (uint32_t col_tile_id = shard_col_start_id; col_tile_id < shard_col_end_id;
                 col_tile_id += num_pages_per_packet) {
                uint32_t tile_id = row_tile_id * in_col_tiles + col_tile_id;
                dfb.reserve_back(num_pages_per_packet);

                uint32_t num_pages_to_read = std::min(shard_col_end_id - col_tile_id, num_pages_per_packet);
                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    noc_obj.async_read(
                        input_tensor_addrgen, dfb, page_size, {.page_id = tile_id}, {.offset_bytes = j * page_size});
                    tile_id++;
                }

                noc_obj.async_read_barrier();
                dfb.push_back(num_pages_per_packet);
            }
        }
    } else {
        // Copy from intermediate buffer to output buffer
        // Compute where remote sender dumped data into intermediate buffer.
        // Should follow same logic as sender writer.

        const uint32_t sender_relative_ring_id =
            (remote_device_ring_id < my_ring_id) ? remote_device_ring_id : remote_device_ring_id - 1;

        uint32_t packet_id = 0;

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                dfb.reserve_back(num_pages_per_packet);
                size_t l1_write_addr = dfb.get_write_ptr();
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t payload_size_bytes = contig_pages_advanced * page_size;

                // Calculate which chunk we need and wait for it
                uint32_t current_chunk_id = packet_id / chunk_granularity;
                uint32_t wait_chunk_id = current_chunk_id + 1;  // Chunks are 1-based
                // Ensure that current chunk has been sent
                global_semaphore.wait_min(wait_chunk_id);

                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t global_id = sender_relative_ring_id + packet_id * NUM_SENDERS;
                    uint32_t first_id = (global_id % N_DRAM_BANKS) + 2 * N_DRAM_BANKS * (global_id / N_DRAM_BANKS);
                    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                    uint64_t packet_addr =
                        intermediate_tensor_addrgen.get_noc_addr(first_id, 0 /*offset*/, 0 /*noc_id*/);

                    noc_async_read(packet_addr, l1_write_addr, payload_size_bytes);
                    l1_write_addr += payload_size_bytes;
                    packet_id++;
                }
                noc_obj.async_read_barrier();

                dfb.push_back(num_pages_per_packet);
            }
        }
    }

    // Reset global semaphore
    global_semaphore.set(0);
}
