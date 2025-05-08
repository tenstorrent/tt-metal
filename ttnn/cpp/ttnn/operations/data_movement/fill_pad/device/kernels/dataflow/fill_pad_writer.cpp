// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr bool tensor_in_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t fill_value = get_compile_time_arg_val(2);
    const uint32_t element_size_bytes = get_compile_time_arg_val(3);
    uint32_t logical_height = get_compile_time_arg_val(4);
    uint32_t logical_width = get_compile_time_arg_val(5);
    uint32_t padded_height = get_compile_time_arg_val(6);
    uint32_t padded_width = get_compile_time_arg_val(7);
    uint32_t tiles_per_2d_tensor = get_compile_time_arg_val(8);
    uint32_t tiles_per_tile_row = get_compile_time_arg_val(9);
    // hardware constraints
    constexpr uint32_t tile_size = get_compile_time_arg_val(10);
    constexpr uint32_t tile_hw = tile_size * tile_size;
    constexpr uint32_t face_size = get_compile_time_arg_val(11);
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t alignment_adjustor = 16;

    uint32_t rt_arg_ind = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t cb_page_size = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t starting_tile_offset = get_arg_val<uint32_t>(rt_arg_ind++);
    uint32_t num_2d_tensors = get_arg_val<uint32_t>(rt_arg_ind++);

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(12),   // Memory layout
        get_compile_time_arg_val(13),   // The number of sharding cores
        get_compile_time_arg_val(14),   // The page size we offset each write to
        get_compile_time_arg_val(15),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(16),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(17),   // pages_per_shard_x
        get_compile_time_arg_val(18)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_ind));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = dst_addr, .shard_array = mapping_table};
#else
    const DataFormat data_format = get_dataformat(cb_id_0);
    const InterleavedAddrGenFast<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = tile_hw * element_size_bytes,
        .data_format = data_format  // page_size needs to be tile_size_bytes
    };
#endif

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_0, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_id_0);
    volatile tt_l1_ptr uint32_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    for (uint32_t i = 0; i < cb_page_size; i++) {
        l1_ptr[i] = fill_value;
    }
    cb_push_back(cb_id_0, 1);

    auto fill_pad_2d_tensor = [&](const uint32_t& tile_offset) {
        uint32_t start_col;
        for (uint32_t row = 0; row < padded_height; row++) {
            if (row < logical_height) {
                start_col = logical_width;
            } else {
                start_col = 0;
            }
            uint32_t curr_tile = (row / tile_size) * tiles_per_tile_row + (start_col / tile_size) + tile_offset;
            uint32_t r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
            uint32_t c_f_offset = ((start_col % tile_size) / face_size) * face_hw + (start_col % face_size);
            uint32_t face_offset = r_f_offset + c_f_offset;

            for (uint32_t col = start_col; col < padded_width;) {
                // so for each iteration of col, we will be writing at most 2 faces
                uint64_t start_tile_noc_addr = get_noc_addr(curr_tile, s0);
                uint32_t face = face_offset / (face_hw);

                uint64_t dst_noc_addr = start_tile_noc_addr + face_offset * element_size_bytes;
                uint32_t alignment_offset = dst_noc_addr % alignment_adjustor;
                uint32_t elems_to_write = col % face_size == 0 ? face_size : face_size - (col % face_size);
                uint32_t bytes_to_write = elems_to_write * element_size_bytes;
                noc_async_write(l1_write_addr + alignment_offset, dst_noc_addr, bytes_to_write);
                col += elems_to_write;
                face_offset += elems_to_write;

                if (face % 2 == 0) {
                    face_offset += face_size * (face_size - 1);
                } else {
                    curr_tile++;
                    face_offset -= face_size * (face_size + 1);
                }
            }
        }
    };

    for (uint32_t t = 0; t < num_2d_tensors; t++) {
        fill_pad_2d_tensor(t * tiles_per_2d_tensor + starting_tile_offset);
    }
    noc_async_write_barrier();
}
