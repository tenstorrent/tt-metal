// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // get input tensor DRAM and find starting points for pad iteration
    const std::uint32_t tensor_dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    // const std::uint32_t beginning_row = get_arg_val<uint32_t>(1);
    // const std::uint32_t beginning_col = get_arg_val<uint32_t>(2);

    // hardware constraints
    constexpr uint32_t face_size = 16;
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t tile_size = 32;
    constexpr uint32_t tile_hw = tile_size * tile_size;

    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr bool tensor_in_dram = get_compile_time_arg_val(1) == 1;
    const std::uint32_t fill_value = get_compile_time_arg_val(4);
    const std::uint32_t element_size_bytes = get_compile_time_arg_val(5);

    const auto tensor = get_interleaved_addr_gen<tensor_in_dram, 16>(tensor_buffer_src_addr);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t cb_page_size = get_arg_val<uint32_t>(1);
    uint32_t logical_height = get_arg_val<uint32_t>(2);
    uint32_t logical_width = get_arg_val<uint32_t>(3);
    uint32_t padded_height = get_arg_val<uint32_t>(4);
    uint32_t padded_width = get_arg_val<uint32_t>(5);

    uint32_t tiles_per_tile_row = (logical_width + tile_size - 1) / tile_size;

#define dst_stick_size_is_pow2 get_compile_time_arg_val(2) == 1
#if (dst_stick_size_is_pow2)
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(3);
    const InterleavedPow2AddrGen<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = dst_log_base_2_of_page_size  // Needs to be log2(tile_size)
    };
#else
    const InterleavedAddrGen<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr, .page_size = tile_hw * element_size_bytes  // needs to be tile_size
    };
#endif

    const InterleavedAddrGenFast<true> tiled_addr_gen = {
        .bank_base_address = input_buffer_src_addr,
        .page_size = tile_size_bytes,
        .data_format = data_format,
    };

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_0, 1);
    uint32_t l1_write_addr = get_write_ptr(cb_id_0);
    volatile tt_l1_ptr uint32_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    // *l1_ptr = fill_value;
    std::memset(reinterpret_cast<void*>(const_cast<uint32_t*>(l1_ptr)), fill_value, cb_page_size * sizeof(uint32_t));
    cb_push_back(cb_id_0, 1);  // Push the fill value to the circular buffer once

    uint32_t start_col;
    for (uint32_t row = 0; row < padded_height; row++) {
        if (row < logical_height) {
            start_col = logical_width;
        } else {
            start_col = 0;
        }

        uint32_t curr_tile = (row / tile_size) * tiles_per_tile_row + (start_col / tile_size);
        uint32_t r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
        uint32_t c_f_offset = ((start_col % tile_size) / face_size) * face_hw + (start_col % face_size);
        uint32_t face_offset = r_f_offset + c_f_offset;

        for (uint32_t col = start_col; col < padded_width;) {
            // so for each iteration of col, we will be writing at most 2 faces
            // use addrgen to get address of start of tile and use face_offset to write to the correct location
            // but how do I get the address of the start of the tile?
            uint64_t dst_noc_addr = get_noc_addr(curr_tile, s0) + face_offset * element_size_bytes;
            uint32_t alignment_offset = dst_noc_addr % 16;
            uint32_t elems_to_write = col % face_size == 0 ? face_size : face_size - (col % face_size);
            uint32_t bytes_to_write = elems_to_write * element_size_bytes;
            noc_async_write(l1_read_addr + alignment_offset, dst_noc_addr, bytes_to_write);
            noc_async_write_barrier();
            col += elems_to_write;
            face_offset += elems_to_write;

            uint32_t face = face_offset / (face_hw);
            if (face % 2 == 0) {
                face_offset += face_size * face_size;
            } else {
                curr_tile++;
                face_offset -= face_size * face_size;
            }
        }
    }
}
