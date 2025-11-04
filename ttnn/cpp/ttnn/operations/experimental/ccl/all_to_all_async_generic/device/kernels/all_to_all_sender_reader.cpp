// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ckernel.h"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
constexpr uint32_t current_device_id = get_compile_time_arg_val(2);
constexpr uint32_t num_devices = get_compile_time_arg_val(3);
constexpr uint32_t outer_dims_size = get_compile_time_arg_val(4);
constexpr uint32_t split_dim_size = get_compile_time_arg_val(5);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(6);
constexpr uint32_t last_dim_size = get_compile_time_arg_val(7);
constexpr uint32_t number_pages_per_packet = get_compile_time_arg_val(8);
constexpr uint32_t has_reader_tail = get_compile_time_arg_val(9);
constexpr uint32_t has_writer_tail = get_compile_time_arg_val(10);

template <typename AddrGenType>
void read_data(
    uint32_t device_id, uint32_t tile_id, uint32_t l1_write_addr, AddrGenType input_addrgen, uint32_t i, bool last) {
    // half tile output case: add padding to the last tile
    bool pad_last_half_tile = has_reader_tail && last;
    if (pad_last_half_tile) {
        noc_async_read(
            input_addrgen.get_noc_addr(tile_id, (device_id % 2) * input_page_size / 2),
            l1_write_addr,
            input_page_size / 2);
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        noc_async_read(zeros_noc_addr, l1_write_addr + input_page_size / 2, input_page_size / 2);
        return;
    }

    // half tile output case: all odd devices write with half tile offset
    bool read_with_half_offset = has_reader_tail && device_id % 2 == 1;
    if (read_with_half_offset) {
        noc_async_read(input_addrgen.get_noc_addr(tile_id, input_page_size / 2), l1_write_addr, input_page_size / 2);
        noc_async_read(
            input_addrgen.get_noc_addr(tile_id + inner_dims_size, 0),
            l1_write_addr + input_page_size / 2,
            input_page_size / 2);
        return;
    }

    // half tile input case: odd devices need to read with half tile offset starting from the second tile in block
    bool read_with_half_offset_for_writer = has_writer_tail && current_device_id % 2 == 1 && i / last_dim_size > 0;
    if (read_with_half_offset_for_writer) {
        noc_async_read(
            input_addrgen.get_noc_addr(tile_id - last_dim_size, input_page_size / 2),
            l1_write_addr,
            input_page_size / 2);
        noc_async_read(
            input_addrgen.get_noc_addr(tile_id, 0), l1_write_addr + input_page_size / 2, input_page_size / 2);
        return;
    }

    noc_async_read_tile(tile_id, input_addrgen, l1_write_addr);
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t input_address = get_arg_val<address_t>(arg_idx++);
    constexpr auto input_tensor_args = TensorAccessorArgs<11>();
    auto input_addrgen = TensorAccessor(input_tensor_args, input_address, input_page_size);
    constexpr uint32_t split_num_half_tiles = split_dim_size * 2 / num_devices;
#define LINEAR 1
#define RING 2

#if TOPOLOGY == LINEAR
    for (uint32_t device_id = 0; device_id < num_devices; ++device_id) {
#elif TOPOLOGY == RING
    for (int d = num_devices - 1; d >= 0; --d) {
        int distance = (d + 1) / 2;
        int val = (d % 2 == 0) ? distance : -distance;
        uint32_t device_id = (current_device_id + val + num_devices) % num_devices;
#else
#error "Unsupported Topology Type"
#endif
        const uint32_t device_read_offset = (split_num_half_tiles * device_id) / 2;
        const uint32_t split_num_tiles = (split_num_half_tiles + 1) / 2;
        uint64_t block_size = outer_dims_size * split_num_tiles * inner_dims_size;

        auto calculate_tile = [&](int b) {
            const uint32_t o = b / (split_num_tiles * inner_dims_size);
            const uint32_t s = (b / inner_dims_size) % split_num_tiles;
            const uint32_t i = b % inner_dims_size;
            const uint32_t tile_id =
                o * inner_dims_size * split_dim_size + device_read_offset * inner_dims_size + s * inner_dims_size + i;
            return std::tuple{s, i, tile_id};
        };

        for (uint64_t block_idx = 0; block_idx < block_size; block_idx += number_pages_per_packet) {
            uint32_t tiles_this_iteration = std::min(number_pages_per_packet, uint32_t(block_size - block_idx));

            cb_reserve_back(cb0_id, tiles_this_iteration);
            address_t l1_write_addr = get_write_ptr(cb0_id);
            for (uint32_t t = 0; t < tiles_this_iteration; ++t) {
                auto [split_idx, inner_idx, tile_id] = calculate_tile(block_idx + t);
                read_data(
                    device_id,
                    tile_id,
                    l1_write_addr + t * input_page_size,
                    input_addrgen,
                    inner_idx,
                    split_idx == split_num_tiles - 1);
            }

            noc_async_read_barrier();
            cb_push_back(cb0_id, tiles_this_iteration);
        }
    }
}
