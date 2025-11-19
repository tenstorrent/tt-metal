// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/dprint_pages.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t src_num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t dst_num_tiles = get_arg_val<uint32_t>(3);
    const uint32_t dst_shard_width = get_arg_val<uint32_t>(4);
    const uint32_t nD_stride = get_arg_val<uint32_t>(5);
    const uint32_t d_stride = get_arg_val<uint32_t>(6);
    const uint32_t n_stride = get_arg_val<uint32_t>(7);
    const uint32_t c_stride = get_arg_val<uint32_t>(8);
    const uint32_t D = get_arg_val<uint32_t>(9);
    const uint32_t N = get_arg_val<uint32_t>(10);
    const uint32_t C = get_arg_val<uint32_t>(11);
    const uint32_t Ht = get_arg_val<uint32_t>(12);
    const uint32_t Wt = get_arg_val<uint32_t>(13);
    const uint32_t cND = get_arg_val<uint32_t>(14);  // collapsed dims > 5
    const uint32_t src_addr_b = get_arg_val<uint32_t>(15);
    const uint32_t nD_stride_b = get_arg_val<uint32_t>(16);
    const uint32_t d_stride_b = get_arg_val<uint32_t>(17);
    const uint32_t n_stride_b = get_arg_val<uint32_t>(18);
    const uint32_t c_stride_b = get_arg_val<uint32_t>(19);
    const uint32_t src_num_tiles_b = get_arg_val<uint32_t>(20);
    const uint32_t current_row = get_arg_val<uint32_t>(21);
    const uint32_t num_rows = get_arg_val<uint32_t>(22);
    uint32_t page_size = get_arg_val<uint32_t>(23);
    // remove the extra tile based args here

    constexpr auto cb_id_src = tt::CBIndex::c_0;
    constexpr auto cb_id_src_b = tt::CBIndex::c_1;

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

#if SRC_SHARDED
    cb_reserve_back(cb_id_src, src_num_tiles);  // we are not usign src_num_tiels , how do we do this for shar
    // sharded then ? , think
    cb_push_back(cb_id_src, src_num_tiles);
#else
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const uint32_t aligned_page_size = ((page_size + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;

    DPRINT << "READER:" << "src_addr: " << src_addr << "src_tile_bytes: " << src_tile_bytes
           << "page_size: " << page_size << "Aligend page size: " << aligned_page_size << ENDL();

    const auto src = TensorAccessor(src_args, src_addr, aligned_page_size);

    const uint32_t tile_hw = get_tile_hw(cb_id_src);  // Number of elements
    const uint32_t element_size = src_tile_bytes / tile_hw;

#endif
#if SRC_SHARDED_B
    cb_reserve_back(cb_id_src_b, src_num_tiles_b);
    cb_push_back(cb_id_src_b, src_num_tiles_b);
#else
    const uint32_t src_tile_bytes_b = get_tile_size(cb_id_src_b);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b, aligned_page_size);
#endif

    constexpr uint32_t onetile = 1;
    constexpr bool has_sharding = get_compile_time_arg_val(src_args.next_compile_time_args_offset()) == 1;

    uint32_t num_tiles_read = 0;

    DPRINT << "num_rows: " << num_rows << ", tile_hw: " << tile_hw << ", num_tiles_read: " << num_tiles_read
           << ", dst_num_tiles :" << dst_num_tiles << ", src_tile_bytes: " << src_tile_bytes
           << ", element_size: " << element_size << ENDL();

    auto row_width = aligned_page_size / element_size;

    const uint32_t div = (row_width + tile_hw - 1) / tile_hw;  // ceil of row_widht / tile_hw

    auto max_elements = row_width > tile_hw ? row_width : tile_hw;
    DPRINT << "current row: " << current_row << "waiting for reserve" << ENDL();

    uint32_t bytes_to_read = aligned_page_size > src_tile_bytes ? src_tile_bytes : aligned_page_size;
    // alignment issues in doign this ?

    for (uint32_t t = 0; t < div; t++) {
        cb_reserve_back(cb_id_src, 1);
        cb_reserve_back(cb_id_src_b, 1);

        uint32_t l1_write_addr_src = get_write_ptr(cb_id_src);
        uint32_t l1_write_addr_original =
            l1_write_addr_src;  // Save the original address, does get_write_ptr return same ptr ?

        uint32_t l1_write_addr_src_b = get_write_ptr(cb_id_src_b);
        uint32_t l1_write_addr_original_b = l1_write_addr_src_b;

        volatile tt_l1_ptr float* ptr = reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_addr_src);
        DPRINT << "Tile: " << t << ", Printing the tile values A  (before) " << ENDL();
        for (uint32_t j = 0; j < tile_hw; j++) {
            ptr[j] = 0.0;
        }
        for (uint32_t j = 0; j < tile_hw; j++) {
            DPRINT << ptr[j] << " ";
        }
        DPRINT << ENDL();

        volatile tt_l1_ptr float* ptr_b = reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_addr_src_b);
        DPRINT << "Tile: " << t << ", Printing the tile values B  (before) " << ENDL();
        for (uint32_t j = 0; j < max_elements; j++) {
            ptr_b[j] = 0.0;
        }
        for (uint32_t j = 0; j < max_elements; j++) {
            DPRINT << ptr_b[j] << " ";
        }
        DPRINT << ENDL();

        for (uint32_t i = 0; i < num_rows; i++) {  // num of rows per tile, if row is bigger than tile size it is 1 here

            uint64_t src_noc_addr = get_noc_addr(current_row + i, src) + bytes_to_read * t;
            noc_async_read(
                src_noc_addr,
                l1_write_addr_src,
                bytes_to_read);  // Read only 16 bytes , multiepl of 16 for aligend and optimal performance ?

            uint64_t src_noc_addr_b = get_noc_addr(current_row + i, src_b) + bytes_to_read * t;
            noc_async_read(src_noc_addr_b, l1_write_addr_src_b, bytes_to_read);

            l1_write_addr_src += bytes_to_read;
            l1_write_addr_src_b += bytes_to_read;
        }
        noc_async_read_barrier();

        DPRINT << "Tile: " << t << ", Printing the tile values A  (after) " << ENDL();
        volatile tt_l1_ptr float* ptr_a_after = reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_addr_original);
        for (uint32_t j = 0; j < tile_hw; j++) {
            DPRINT << ptr_a_after[j] << " ";
        }
        DPRINT << ENDL();

        DPRINT << "Tile: " << t << ", Printing the tile values B  (after) " << ENDL();
        volatile tt_l1_ptr float* ptr_b_after = reinterpret_cast<volatile tt_l1_ptr float*>(l1_write_addr_original_b);
        for (uint32_t j = 0; j < tile_hw; j++) {
            DPRINT << ptr_b_after[j] << " ";
        }
        DPRINT << ENDL();

        cb_push_back(cb_id_src, 1);
        cb_push_back(cb_id_src_b, 1);
    }

    DPRINT << "Reader Exit" << ENDL();
}
