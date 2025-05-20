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

void kernel_main() {
    size_t arg_idx = 0;

    address_t output_buffer_addr = get_arg_val<address_t>(arg_idx++);
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
    auto output_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = output_buffer_addr, .page_size = page_size, .data_format = get_dataformat(cb_id)};

    if (my_ring_id == remote_device_ring_id) {
        // Follows same logic as sender reader for local copy.
        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                cb_wait_front(cb_id, num_pages_per_packet);
                size_t l1_read_addr = get_read_ptr(cb_id);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
                constexpr uint32_t payload_size_bytes = contig_pages_advanced * page_size;
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t col_tile = out_col_id + j;
                    uint32_t tile_id = out_row_id * out_col_tiles + col_tile;
                    noc_async_write_tile(tile_id, output_tensor_addrgen, l1_read_addr);

                    l1_read_addr += payload_size_bytes;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_id, num_pages_per_packet);
            }
        }

    } else {
        // Copy from intermediate buffer to output buffer
        // Compute where remote sender dumped data into intermediate buffer.
        // Should follow same logic as sender writer.

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                cb_wait_front(cb_id, num_pages_per_packet);
                size_t l1_read_addr = get_read_ptr(cb_id);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t contig_pages_advanced = 1;  // always write 1 tile at a time to output
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t col_tile = out_col_id + j;
                    uint32_t tile_id = out_row_id * out_col_tiles + col_tile;
                    noc_async_write_tile(tile_id, output_tensor_addrgen, l1_read_addr);
                    l1_read_addr += page_size;
                }
                noc_async_writes_flushed();

                cb_pop_front(cb_id, num_pages_per_packet);
            }
        }
    }
    noc_async_write_barrier();
}
