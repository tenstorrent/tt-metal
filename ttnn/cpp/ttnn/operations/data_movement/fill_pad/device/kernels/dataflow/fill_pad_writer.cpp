// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

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
                DPRINT << input_l1_ptr[i + r_f_offset + r_offset] << ", ";
            }
            DPRINT << "] | ";
            DPRINT << "[";
            for (uint32_t i = 0; i < face_size; i++) {
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

void print_tile_bf16(volatile tt_l1_ptr uint16_t* input_l1_ptr, const uint32_t face_size, uint32_t row_stop = 32) {
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
#define SHARDED get_compile_time_arg_val(12) == 1
    constexpr uint32_t face_hw = face_size * face_size;
    constexpr uint32_t alignment_adjustor = 16;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t cb_page_size = get_arg_val<uint32_t>(1);
    uint32_t starting_tile_offset = get_arg_val<uint32_t>(2);
    uint32_t num_2d_tensors = get_arg_val<uint32_t>(3);

#if (SHARDED)
    typedef Sharded_Info<
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(17),
        get_compile_time_arg_val(18),
        get_compile_time_arg_val(19)>
        tensor_shard_info;

    std::pair<const mapping_table_t*, const uint32_t> map =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(4));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = dst_addr, .shard_array = map.first};
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

    cb_reserve_back(1, 1);
    // uint32_t l1_read_addr = get_write_ptr(1);
    uint16_t l1_read_addr = get_write_ptr(1);
    // volatile tt_l1_ptr uint32_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);
    volatile tt_l1_ptr uint16_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr);

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
                noc_async_read(start_tile_noc_addr, l1_read_addr, tile_hw * element_size_bytes);
                noc_async_read_barrier();
                DPRINT << "curr_tile: " << curr_tile << ENDL();
                DPRINT << "Writing " << elems_to_write << " elements to tile " << curr_tile << " with face offset "
                       << face_offset << ENDL();
                // print_tile(input_l1_ptr, face_size, row);
                print_tile_bf16(input_l1_ptr, face_size, row);
                noc_async_write(l1_write_addr + alignment_offset, dst_noc_addr, bytes_to_write);
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

    for (uint32_t t = 0; t < num_2d_tensors; t++) {
        fill_pad_2d_tensor(t * tiles_per_2d_tensor + starting_tile_offset);
    }
}
