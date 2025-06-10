// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_ring_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t in_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t in_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_row_device_stride = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_col_device_stride = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_offset = get_arg_val<uint32_t>(arg_idx++);

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    bool cur_is_forward = num_targets_forward_direction > num_targets_backward_direction;
    uint32_t forward_hops = 1;
    uint32_t backward_hops = 1;
    uint32_t dst_ring_id;
    for (uint32_t i = 0; i < ring_size - 1; ++i) {
        // Switch direction when we've reached the end of the forward or backward direction
        if (forward_hops == num_targets_forward_direction + 1) {
            cur_is_forward = false;
        }
        if (backward_hops == num_targets_backward_direction + 1) {
            cur_is_forward = true;
        }

        if (cur_is_forward) {
            dst_ring_id = (my_ring_id + forward_hops) % ring_size;
            forward_hops++;
        } else {
            dst_ring_id = (my_ring_id - backward_hops + ring_size) % ring_size;
            backward_hops++;
        }

        uint32_t shard_row_start_id = dst_ring_id * input_row_device_stride;
        uint32_t shard_col_start_id = dst_ring_id * input_col_device_stride;
        uint32_t shard_row_end_id = shard_row_start_id + input_shard_row_tiles;
        uint32_t shard_col_end_id = shard_col_start_id + input_shard_col_tiles;

        for (uint32_t row_tile_id = shard_row_start_id; row_tile_id < shard_row_end_id; row_tile_id++) {
            for (uint32_t col_tile_id = shard_col_start_id; col_tile_id < shard_col_end_id;
                 col_tile_id += packet_size_in_pages) {
                uint32_t tile_id = row_tile_id * in_col_tiles + col_tile_id;
                cb_reserve_back(cb0_id, packet_size_in_pages);
                const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
                uint32_t l1_write_addr = l1_write_addr_base;

                uint32_t num_pages_to_read = std::min(shard_col_end_id - col_tile_id, packet_size_in_pages);
                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    noc_async_read_tile(tile_id, tensor0_addrgen, l1_write_addr);
                    l1_write_addr += tensor0_page_size;
                    tile_id++;
                }

                noc_async_read_barrier();
                cb_push_back(cb0_id, packet_size_in_pages);
            }
        }
    }
}
