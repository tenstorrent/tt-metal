// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void print_tile(volatile tt_l1_ptr uint32_t* input_l1_ptr, const uint32_t face_size, uint32_t row_stop = 32) {
    uint32_t row_count = 0;
    for (uint32_t r_f = 0; r_f < 2; r_f++) {
        for (uint32_t i = 0; i < 105; i++) {
            DPRINT << "_";
        }
        DPRINT << ENDL();
        uint32_t r_f_offset = r_f * 2 * face_size * face_size;
        for (uint32_t r = 0; r < face_size; r++) {
            uint32_t r_offset = r * face_size;
            DPRINT << "[";
            for (uint32_t i = 0; i < face_size; i++) {
                DPRINT << BF16(input_l1_ptr[i + r_f_offset + r_offset]) << ", ";
            }
            DPRINT << "] | ";
            DPRINT << "[";
            for (uint32_t i = 0; i < face_size; i++) {
                DPRINT << BF16(input_l1_ptr[i + r_f_offset + r_offset + (face_size * face_size)]) << ", ";
            }
            DPRINT << "]" << ENDL();
            row_count++;
            if (row_count == row_stop) {
                return;
            }
        }
        for (uint32_t i = 0; i < 105; i++) {
            DPRINT << "_";
        }
        DPRINT << ENDL();
    }
}

void print_face_row(volatile tt_l1_ptr uint32_t* input_l1_ptr) {
    DPRINT << "[";
    for (uint32_t i = 0; i < 16; i++) {
        // DPRINT << BF16(input_l1_ptr[i]) << ", ";
        DPRINT << input_l1_ptr[i] << ", ";
    }
    DPRINT << "]" << ENDL();
}

void kernel_main() {
    // get input tensor DRAM and find starting points for pad iteration
    const std::uint32_t tensor_buffer_src_addr = get_arg_val<uint32_t>(0);
    // const std::uint32_t beginning_row = get_arg_val<uint32_t>(1);
    // const std::uint32_t beginning_col = get_arg_val<uint32_t>(2);

    // hardware constraints
    constexpr uint32_t face_size = 16;
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t tile_size = 32;
    constexpr uint32_t tile_hw = tile_size * tile_size;

    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_1 = get_compile_time_arg_val(1);
    constexpr bool tensor_in_dram = get_compile_time_arg_val(2) == 1;
#define BFP16 get_compile_time_arg_val(7) == 1
#if (BFP16)
    const uint16_t fill_value = get_compile_time_arg_val(5);
#else
    const std::uint32_t fill_value = get_compile_time_arg_val(5);
#endif
    const std::uint32_t element_size_bytes = get_compile_time_arg_val(6);

    const auto tensor = get_interleaved_addr_gen<tensor_in_dram, 16>(tensor_buffer_src_addr);

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t cb_page_size = get_arg_val<uint32_t>(1);
    uint32_t logical_height = get_arg_val<uint32_t>(2);
    uint32_t logical_width = get_arg_val<uint32_t>(3);
    uint32_t padded_height = get_arg_val<uint32_t>(4);
    uint32_t padded_width = get_arg_val<uint32_t>(5);

    uint32_t tiles_per_tile_row = (logical_width + tile_size - 1) / tile_size;  // do this in program factory

#define dst_stick_size_is_pow2 get_compile_time_arg_val(3) == 1
#if (dst_stick_size_is_pow2)
    constexpr uint32_t dst_log_base_2_of_page_size = get_compile_time_arg_val(4);
    const InterleavedPow2AddrGen<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = dst_log_base_2_of_page_size  // Needs to be log2(tile_size)
    };
#else
    const InterleavedAddrGen<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr, .page_size = tile_hw * element_size_bytes  // needs to be tile_size
    };
#endif
    const uint32_t tile_size_bytes = get_tile_size(cb_id_1);
    const DataFormat data_format = get_dataformat(cb_id_1);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dst_addr,
        .page_size = 32 * 32 * element_size_bytes,
        .data_format = data_format,
    };

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_0, 1);
#if (BFP16)
    uint16_t l1_write_addr = get_write_ptr(cb_id_0);
    volatile tt_l1_ptr uint16_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    for (uint32_t i = 0; i < cb_page_size; i++) {
        l1_ptr[i] = fill_value;
    }
#else
    uint32_t l1_write_addr = get_write_ptr(cb_id_0);
    volatile tt_l1_ptr uint32_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_write_addr);
    for (uint32_t i = 0; i < cb_page_size; i++) {
        l1_ptr[i] = fill_value;
    }
#endif
    cb_push_back(cb_id_0, 1);  // Push the fill value to the circular buffer once

    cb_reserve_back(cb_id_1, 1);
    uint32_t l1_read_addr = get_write_ptr(cb_id_1);
    volatile tt_l1_ptr uint32_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);

    // DPRINT << "logical height: " << logical_height << ENDL();
    // DPRINT << "logical width: " << logical_width << ENDL();
    // DPRINT << "Padded height: " << padded_height << ENDL();
    // DPRINT << "Padded width: " << padded_width << ENDL();
    // x, x, x, x, x, 0, 0, 0
    // 0, 0, 0, 0, 0, 0, 0, 0,
    uint32_t start_col;
    for (uint32_t row = 0; row < padded_height; row++) {
        if (row < logical_height) {
            start_col = logical_width;
        } else {
            start_col = 0;
        }
        DPRINT << "Row: " << row << " Start col: " << start_col << ENDL();
        uint32_t curr_tile = (row / tile_size) * tiles_per_tile_row + (start_col / tile_size);
        uint32_t r_f_offset = ((row % tile_size) / face_size) * 2 * face_hw + (row % face_size) * face_size;
        uint32_t c_f_offset = ((start_col % tile_size) / face_size) * face_hw + (start_col % face_size);
        uint32_t face_offset = r_f_offset + c_f_offset;

        for (uint32_t col = start_col; col < padded_width;) {
            // so for each iteration of col, we will be writing at most 2 faces
            uint64_t start_tile_noc_addr = get_noc_addr(curr_tile, s0);
            uint32_t face = face_offset / (face_hw);

            uint64_t dst_noc_addr = start_tile_noc_addr + face_offset * element_size_bytes;
            uint32_t alignment_offset = dst_noc_addr % 16;
            uint32_t elems_to_write = col % face_size == 0 ? face_size : face_size - (col % face_size);
            uint32_t bytes_to_write = elems_to_write * element_size_bytes;
            noc_async_read(
                start_tile_noc_addr + ((face_offset / 16) * 16) * element_size_bytes,
                l1_read_addr,
                16 * element_size_bytes);
            noc_async_read_barrier();
            DPRINT << "Reading " << 16 << " elements from tile " << curr_tile
                   << " at face_offset: " << ((face_offset / 16) * 16) << " in face " << face << ENDL();
            print_face_row(input_l1_ptr);
            DPRINT << "Writing " << elems_to_write << " elements to tile " << curr_tile
                   << " at face_offset: " << face_offset << " in face " << face << ENDL();
            noc_async_write(l1_write_addr + alignment_offset, dst_noc_addr, bytes_to_write);
            noc_async_write_barrier();
            col += elems_to_write;
            noc_async_read(
                start_tile_noc_addr + ((face_offset / 16) * 16) * element_size_bytes,
                l1_read_addr,
                16 * element_size_bytes);
            noc_async_read_barrier();
            print_face_row(input_l1_ptr);
            face_offset += elems_to_write;

            if (face % 2 == 0) {
                face_offset += face_size * (face_size - 1);
            } else {
                // noc_async_read(start_tile_noc_addr, l1_read_addr, 32*32*element_size_bytes);
                noc_async_read_tile(curr_tile, s, l1_read_addr);
                noc_async_read_barrier();
                print_tile(input_l1_ptr, 16, row + 1);
                curr_tile++;
                face_offset -= face_size * face_size;
            }
            DPRINT << ENDL();
        }
    }
}
