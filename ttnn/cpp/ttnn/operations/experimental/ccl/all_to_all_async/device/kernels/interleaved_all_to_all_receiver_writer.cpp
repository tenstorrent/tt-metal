// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
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

void kernel_main() {
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

    auto output_tensor_addrgen = TensorAccessor(tensor::output);

    Noc noc_obj;
    DataflowBuffer dfb(dfb::receiver_data);

    if (my_ring_id == remote_device_ring_id) {
        // Follows same logic as sender reader for local copy.
        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                dfb.wait_front(num_pages_per_packet);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t contig_pages_advanced = 1;  // always 1 for interleaved
                constexpr uint32_t payload_size_bytes = contig_pages_advanced * page_size;
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t col_tile = out_col_id + j;
                    uint32_t tile_id = out_row_id * out_col_tiles + col_tile;
                    noc_obj.async_write(
                        dfb,
                        output_tensor_addrgen,
                        payload_size_bytes,
                        {.offset_bytes = j * payload_size_bytes},
                        {.page_id = tile_id});
                }
                noc_obj.async_writes_flushed();
                dfb.pop_front(num_pages_per_packet);
            }
        }

    } else {
        // Copy from intermediate buffer to output buffer
        // Compute where remote sender dumped data into intermediate buffer.
        // Should follow same logic as sender writer.

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += num_pages_per_packet) {
                dfb.wait_front(num_pages_per_packet);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, num_pages_per_packet);

                constexpr uint32_t contig_pages_advanced = 1;  // always write 1 tile at a time to output
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    uint32_t col_tile = out_col_id + j;
                    uint32_t tile_id = out_row_id * out_col_tiles + col_tile;
                    noc_obj.async_write(
                        dfb, output_tensor_addrgen, page_size, {.offset_bytes = j * page_size}, {.page_id = tile_id});
                }
                noc_obj.async_writes_flushed();

                dfb.pop_front(num_pages_per_packet);
            }
        }
    }
    noc_obj.async_write_barrier();
}
