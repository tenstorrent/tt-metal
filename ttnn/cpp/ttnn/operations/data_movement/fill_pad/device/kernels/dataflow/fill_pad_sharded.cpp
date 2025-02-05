// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
                // DPRINT << BF16(input_l1_ptr[i + r_f_offset + r_offset]) << ", ";
                DPRINT << input_l1_ptr[i + r_f_offset + r_offset] << ", ";
            }
            DPRINT << "] | ";
            DPRINT << "[";
            for (uint32_t i = 0; i < face_size; i++) {
                // DPRINT << BF16(input_l1_ptr[i + r_f_offset + r_offset + (face_size * face_size)]) << ", ";
                DPRINT << input_l1_ptr[i + r_f_offset + r_offset + (face_size * face_size)] << ", ";
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
    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_1 = get_compile_time_arg_val(1);
    constexpr bool tensor_in_dram = get_compile_time_arg_val(2) == 1;
    const uint32_t fill_value = get_compile_time_arg_val(3);
    const uint32_t element_size_bytes = get_compile_time_arg_val(4);
    uint32_t logical_height = get_compile_time_arg_val(5);
    uint32_t logical_width = get_compile_time_arg_val(6);
    uint32_t padded_height = get_compile_time_arg_val(7);
    uint32_t padded_width = get_compile_time_arg_val(8);
    uint32_t tiles_per_2d_tensor = get_compile_time_arg_val(9);
    uint32_t tiles_per_tile_row = get_compile_time_arg_val(10);
    // hardware constraints
    constexpr uint32_t tile_size = get_compile_time_arg_val(11);
    constexpr uint32_t tile_hw = tile_size * tile_size;
    constexpr uint32_t face_size = get_compile_time_arg_val(12);
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t alignment_adjustor = 16;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t cb_page_size = get_arg_val<uint32_t>(1);
    uint32_t starting_tile_offset = get_arg_val<uint32_t>(2);
    uint32_t num_2d_tensors = get_arg_val<uint32_t>(3);

    const DataFormat data_format = get_dataformat(cb_id_0);
    const InterleavedAddrGenFast<tensor_in_dram> s0 = {
        .bank_base_address = dst_addr,
        .page_size = tile_hw * element_size_bytes,
        .data_format = data_format  // page_size needs to be tile_size_bytes
    };

    // Reserve and push the fill value into the circular buffer
    cb_reserve_back(cb_id_1, 1);
    uint32_t l1_fill_addr = get_write_ptr(1);
    volatile tt_l1_ptr uint32_t* l1_fill_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_fill_addr);
    for (uint32_t i = 0; i < cb_page_size; i++) {
        l1_fill_ptr[i] = fill_value;
    }
    cb_push_back(cb_id_1, 1);

    cb_reserve_back(cb_id_0, 1);
    uint32_t l1_read_addr = get_read_ptr(0);
    uint32_t l1_tensor_start = get_read_ptr(0);
    volatile tt_l1_ptr uint32_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);

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

            DPRINT << "row: " << row << " start_col: " << start_col << " padded_width: " << padded_width << ENDL();
            for (uint32_t col = start_col; col < padded_width;) {
                // so for each iteration of col, we will be writing at most 2 faces
                DPRINT << "curr_tile: " << curr_tile << " face_offset: " << face_offset << ENDL();
                uint64_t start_tile_noc_addr = get_noc_addr(
                    l1_tensor_start);  // right now, this line just gets the address of the start of the shard
                noc_async_read(start_tile_noc_addr, l1_read_addr, tile_hw * element_size_bytes);
                noc_async_read_barrier();
                print_tile(input_l1_ptr, face_size, row);
                uint32_t face = face_offset / (face_hw);

                uint64_t dst_noc_addr = start_tile_noc_addr + face_offset * element_size_bytes;
                uint32_t alignment_offset = dst_noc_addr % alignment_adjustor;
                uint32_t elems_to_write = col % face_size == 0 ? face_size : face_size - (col % face_size);
                DPRINT << "elems_to_write: " << elems_to_write << ENDL();
                uint32_t bytes_to_write = elems_to_write * element_size_bytes;
                noc_async_write(l1_fill_addr + alignment_offset, dst_noc_addr, bytes_to_write);
                noc_async_write_barrier();
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

    DPRINT << "num_2d_tensors: " << num_2d_tensors << ENDL();
    DPRINT << "starting_tile_offset: " << starting_tile_offset << ENDL();
    for (uint32_t t = 0; t < num_2d_tensors; t++) {
        DPRINT << "t: " << t << ENDL();
        fill_pad_2d_tensor(t * tiles_per_2d_tensor + starting_tile_offset);
    }
}
