// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_ring_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t chunk_granularity = get_compile_time_arg_val(3);
constexpr uint32_t chunk_num_tiles = get_compile_time_arg_val(4);
constexpr uint32_t num_chunks_per_shard = get_compile_time_arg_val(5);
constexpr uint32_t page_size = get_compile_time_arg_val(6);
constexpr uint32_t cb_id = get_compile_time_arg_val(7);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(8);
constexpr uint32_t N_DRAM_BANKS = get_compile_time_arg_val(9);

constexpr uint32_t NUM_SENDERS = ring_size - 1;

void kernel_main() {
    size_t arg_idx = 0;

    address_t intermediate_buffer_addr = get_arg_val<address_t>(arg_idx++);
    address_t input_buffer_addr = get_arg_val<address_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t in_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t in_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_row_device_stride = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_col_device_stride = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_pages_per_packet = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_device_ring_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    constexpr bool is_dram = true;
    auto input_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = input_buffer_addr, .page_size = page_size, .data_format = get_dataformat(cb_id)};
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = intermediate_buffer_addr, .page_size = page_size, .data_format = get_dataformat(cb_id)};

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
                cb_reserve_back(cb_id, num_pages_per_packet);
                const uint32_t l1_write_addr_base = get_write_ptr(cb_id);
                uint32_t l1_write_addr = l1_write_addr_base;

                uint32_t num_pages_to_read = std::min(shard_col_end_id - col_tile_id, num_pages_per_packet);
                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    noc_async_read_tile(tile_id, input_tensor_addrgen, l1_write_addr);
                    l1_write_addr += page_size;
                    tile_id++;
                }

                noc_async_read_barrier();
                cb_push_back(cb_id, num_pages_per_packet);
            }
        }
    } else {
        // Copy from intermediate buffer to output buffer
        // Compute where remote sender dumped data into intermediate buffer.
        // Should follow same logic as sender writer.

        const uint32_t sender_relative_ring_id =
            (remote_device_ring_id < my_ring_id) ? remote_device_ring_id : remote_device_ring_id - 1;

        volatile tt_l1_ptr uint32_t* global_semaphore_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);
        uint32_t packet_id = 0;

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                cb_reserve_back(cb_id, num_pages_per_packet);
                size_t l1_write_addr = get_write_ptr(cb_id);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t payload_size_bytes = contig_pages_advanced * page_size;

                // Calculate which chunk we need and wait for it
                uint32_t current_chunk_id = packet_id / chunk_granularity;
                uint32_t wait_chunk_id = current_chunk_id + 1;  // Chunks are 1-based
                // Ensure that current chunk has been sent
                noc_semaphore_wait_min(global_semaphore_ptr, wait_chunk_id);

                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t global_id = sender_relative_ring_id + packet_id * NUM_SENDERS;
                    uint32_t first_id = (global_id % N_DRAM_BANKS) + 2 * N_DRAM_BANKS * (global_id / N_DRAM_BANKS);
                    uint64_t packet_addr =
                        get_noc_addr(first_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    noc_async_read(packet_addr, l1_write_addr, payload_size_bytes);
                    l1_write_addr += payload_size_bytes;
                    packet_id++;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id, num_pages_per_packet);
            }
        }
    }

    // Reset global semaphore
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr), 0);
}
