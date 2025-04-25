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

constexpr uint32_t wait_sem_value = 1;

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

    constexpr bool is_dram = true;  // TODO: CT arg
    auto input_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = input_buffer_addr, .page_size = page_size, .data_format = get_dataformat(cb_id)};
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = intermediate_buffer_addr, .page_size = page_size, .data_format = get_dataformat(cb_id)};

    if (my_ring_id == remote_device_ring_id) {
        // TODO: do local copy on this core
        // Follows same logic as sender reader for local copy.
        return;
    } else {
        // Copy from intermediate buffer to output buffer
        // Compute where remote sender dumped data into intermediate buffer.
        // Should follow same logic as sender writer.

        // Wait for semaphore increment from sender. For now, wait until sender is fully done.
        // TODO: Add chunking granularity.
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr) < wait_sem_value);

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                cb_reserve_back(cb_id, num_pages_per_packet);
                size_t l1_write_addr = get_write_ptr(cb_id);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
                // constexpr uint32_t payload_size_bytes = contig_pages_advanced * tensor0_page_size;
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t col_tile = out_col_id + j;
                    uint32_t tile_id = out_row_id * out_col_tiles + col_tile;
                    // uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);
                    noc_async_read_tile(tile_id, intermediate_tensor_addrgen, l1_write_addr);
                    l1_write_addr += page_size;
                }
                noc_async_read_barrier();

                cb_push_back(cb_id, num_pages_per_packet);
            }
        }
    }

    // Reset global semaphore
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr) = 0;
    // DPRINT << "reset done\n";
}
