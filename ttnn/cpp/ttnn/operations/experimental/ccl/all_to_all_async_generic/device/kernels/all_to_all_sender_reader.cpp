// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
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
constexpr uint32_t split_dim_size = get_compile_time_arg_val(4);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(5);
constexpr uint32_t last_dim_size = get_compile_time_arg_val(6);
constexpr uint32_t has_reader_tail = get_compile_time_arg_val(7);
constexpr uint32_t has_writer_tail = get_compile_time_arg_val(8);
constexpr uint32_t concat_num_tiles = get_compile_time_arg_val(9);
constexpr uint32_t dst_inner_dims_size = get_compile_time_arg_val(10);

template <typename AddrGenType>
void read_data(
    const Noc& noc_obj,
    uint32_t device_id,
    uint32_t tile_id,
    uint32_t& l1_write_addr,
    AddrGenType input_addrgen,
    uint32_t c,
    bool last,
    uint32_t payload_size) {
    CoreLocalMem<uint8_t> dst(l1_write_addr);

    // half tile output case: add padding to the last tile
    bool pad_last_half_tile = has_reader_tail && last;
    if (pad_last_half_tile) {
        noc_obj.async_read(
            input_addrgen,
            dst,
            input_page_size / 2,
            {.page_id = tile_id, .offset_bytes = (device_id % 2) * input_page_size / 2},
            {});
        // Device 2.0 migration: legacy primitive retained — MEM_ZEROS_BASE self-read has no typed Noc
        // endpoint today. TODO(#45845): migrate to a LocalL1 endpoint once available.
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        noc_async_read(zeros_noc_addr, l1_write_addr + input_page_size / 2, input_page_size / 2);
        l1_write_addr += input_page_size;
        return;
    }

    // half tile output case: all odd devices write with half tile offset
    bool read_with_half_offset = has_reader_tail && device_id % 2 == 1;
    if (read_with_half_offset) {
        noc_obj.async_read(
            input_addrgen, dst, input_page_size / 2, {.page_id = tile_id, .offset_bytes = input_page_size / 2}, {});
        noc_obj.async_read(
            input_addrgen,
            dst,
            input_page_size / 2,
            {.page_id = tile_id + inner_dims_size},
            {.offset_bytes = input_page_size / 2});
        l1_write_addr += input_page_size;
        return;
    }

    // half tile input case: odd devices need to read with half tile offset starting from the second tile in block
    bool read_with_half_offset_for_writer = has_writer_tail && current_device_id % 2 == 1 && c > 0;
    if (read_with_half_offset_for_writer) {
        noc_obj.async_read(
            input_addrgen,
            dst,
            input_page_size / 2,
            {.page_id = tile_id - last_dim_size, .offset_bytes = input_page_size / 2},
            {});
        noc_obj.async_read(
            input_addrgen, dst, input_page_size / 2, {.page_id = tile_id}, {.offset_bytes = input_page_size / 2});
        l1_write_addr += input_page_size;
        return;
    }

    noc_obj.async_read(input_addrgen, dst, payload_size, {.page_id = tile_id}, {});
    l1_write_addr += payload_size;
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t input_address = get_arg_val<address_t>(arg_idx++);
    uint32_t local_num_devices = get_arg_val<uint32_t>(arg_idx++);
    constexpr auto input_tensor_args = TensorAccessorArgs<11>();
    auto input_addrgen = TensorAccessor(input_tensor_args, input_address);
    constexpr uint32_t split_num_half_tiles = split_dim_size * 2 / num_devices;
    constexpr uint32_t split_num_tiles = (split_num_half_tiles + 1) / 2;
    constexpr bool has_pre_half_tile = has_writer_tail && current_device_id % 2 == 1;
    constexpr bool has_post_half_tile = has_writer_tail && current_device_id % 2 == 0;

    Noc noc_obj;
    DataflowBuffer cb0(cb0_id);

    for (uint32_t did = 0; did < local_num_devices; ++did) {
        int32_t device_offset = get_arg_val<int32_t>(arg_idx++);
        uint32_t device_id = (current_device_id + device_offset + num_devices) % num_devices;

        const uint32_t device_read_offset = (split_num_half_tiles * device_id) / 2;

        auto calculate_tile = [&](int b) {
            const uint32_t o = b / (split_num_tiles * inner_dims_size);
            const uint32_t s = (b / inner_dims_size) % split_num_tiles;
            const uint32_t c = (b / dst_inner_dims_size) % concat_num_tiles;
            const uint32_t i = b % inner_dims_size;
            const uint32_t tile_id =
                o * inner_dims_size * split_dim_size + device_read_offset * inner_dims_size + s * inner_dims_size + i;
            uint32_t payload_size = ((has_pre_half_tile && c == 0) || (has_post_half_tile && c == concat_num_tiles - 1))
                                        ? input_page_size / 2
                                        : input_page_size;
            return std::tuple{s, i, c, tile_id, payload_size};
        };

        uint32_t block_idx = get_arg_val<uint32_t>(arg_idx++);
        uint32_t block_end_id = get_arg_val<uint32_t>(arg_idx++);
        cb0.reserve_back(1);
        address_t l1_write_addr = cb0.get_write_ptr();
        uint32_t current_package_payload = 0;
        uint32_t current_tile = 0;

        while (block_idx < block_end_id) {
            auto [split_idx, inner_idx, concat_idx, tile_id, payload_size] = calculate_tile(block_idx);

            // If package is full, flush and start new package
            if (current_tile == 4 || current_package_payload + payload_size > 2 * input_page_size) {
                noc_obj.async_read_barrier();
                cb0.push_back(1);
                cb0.reserve_back(1);
                l1_write_addr = cb0.get_write_ptr();
                current_package_payload = 0;
                current_tile = 0;
            }

            read_data(
                noc_obj,
                device_id,
                tile_id,
                l1_write_addr,
                input_addrgen,
                concat_idx,
                split_idx == split_num_tiles - 1,
                payload_size);

            current_package_payload += payload_size;
            current_tile++;
            block_idx++;
        }

        // Flush remaining tiles in the last package
        if (current_tile > 0) {
            noc_obj.async_read_barrier();
        }
        cb0.push_back(1);
    }
}
