// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Fill tile with first column for broadcast
FORCE_INLINE void fill_tile_with_first_column(uint32_t cb_id) {
    // Tile with 4 faces (16x16) and 32-bit elements
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));

    constexpr uint32_t num_rows = 16;             // Number of rows per face
    constexpr uint32_t face_row_stride = 16;      // Elements per row
    constexpr uint32_t face_size = 256;           // Total elements per face (16x16)
    constexpr uint32_t face_offset_stride = 512;  // Total elements per pair of faces (2x16x16)

    // Iterate over face pairs (0,1) and (2,3)
    for (uint32_t k = 0, face_offset = 0; k < 2; ++k, face_offset += face_offset_stride) {
        for (uint32_t row = 0, row_offset = 0; row < num_rows; ++row, row_offset += face_row_stride) {
            uint32_t left_dst_offset = face_offset + row_offset;      // Left face (0 or 2)
            uint32_t right_dst_offset = left_dst_offset + face_size;  // Right face (1 or 3)

            // Read the first column value for the current row from the left face
            auto src_val = ptr[left_dst_offset];

            for (uint32_t col = 0; col < face_row_stride; ++col) {
                ptr[left_dst_offset + col] = src_val;   // left face
                ptr[right_dst_offset + col] = src_val;  // right face
            }
        }
    }
}

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src2_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(5);

    // Broadcast flags - compile time args
    constexpr bool bcast_in1 = get_compile_time_arg_val(6) == 1;  // value_true broadcast
    constexpr bool bcast_in2 = get_compile_time_arg_val(7) == 1;  // value_false broadcast

    uint32_t l1_write_addr_in0;
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    uint32_t l1_write_addr_in1;
    uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    DataFormat src1_data_format = get_dataformat(cb_id_in1);
    constexpr bool src1_is_dram = get_compile_time_arg_val(2) == 1;
    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr, .page_size = src1_tile_bytes, .data_format = src1_data_format};

    uint32_t l1_write_addr_in2;
    uint32_t src2_tile_bytes = get_tile_size(cb_id_in2);
    DataFormat src2_data_format = get_dataformat(cb_id_in2);
    constexpr bool src2_is_dram = get_compile_time_arg_val(4) == 1;
    const InterleavedAddrGenFast<src2_is_dram> s2 = {
        .bank_base_address = src2_addr, .page_size = src2_tile_bytes, .data_format = src2_data_format};

    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; tile_id++) {
        // Read predicate (always full)
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(tile_id, s0, l1_write_addr_in0);

        // Read value_true
        cb_reserve_back(cb_id_in1, onetile);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);
        if constexpr (bcast_in1) {
            // For broadcast, read only the first column of this tile
            uint32_t col_tile_id = (tile_id / 32) * 32;  // Get tile ID for first column in row
            noc_async_read_tile(col_tile_id, s1, l1_write_addr_in1);
        } else {
            noc_async_read_tile(tile_id, s1, l1_write_addr_in1);
        }

        // Read value_false
        cb_reserve_back(cb_id_in2, onetile);
        l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        if constexpr (bcast_in2) {
            // For broadcast, read only the first column of this tile
            uint32_t col_tile_id = (tile_id / 32) * 32;  // Get tile ID for first column in row
            noc_async_read_tile(col_tile_id, s2, l1_write_addr_in2);
        } else {
            noc_async_read_tile(tile_id, s2, l1_write_addr_in2);
        }

        noc_async_read_barrier();

        // Apply broadcast fill if needed
        if constexpr (bcast_in1) {
            fill_tile_with_first_column(cb_id_in1);
        }
        if constexpr (bcast_in2) {
            fill_tile_with_first_column(cb_id_in2);
        }

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);
        cb_push_back(cb_id_in2, onetile);
    }
}
