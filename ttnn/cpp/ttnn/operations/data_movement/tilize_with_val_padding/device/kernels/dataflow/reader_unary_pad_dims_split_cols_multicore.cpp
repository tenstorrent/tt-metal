// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, uint32_t val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        // DPRINT << "VAL IS " << val <<ENDL();
        // DPRINT << "BFLOAT6 of val" << BF16((uint16_t)val) <<ENDL();
        ptr[i] = val;
    }
}

void kernel_main() {
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t tile_row_shift_bits = get_compile_time_arg_val(3);
    constexpr uint32_t num_padding_rows = get_compile_time_arg_val(4);
    // DPRINT <<"num padding rows: " << num_padding_rows << ENDL();

    constexpr uint32_t tiles_per_col = get_compile_time_arg_val(5);
    // DPRINT << "tiles_per_col: " << tiles_per_col << ENDL();

    const uint32_t total_num_rows = get_compile_time_arg_val(6);
    // DPRINT << "total_num_rows: " << total_num_rows << ENDL();

    const uint32_t padded_size_per_row = get_compile_time_arg_val(7);
    // DPRINT << "padded_size_per row: " <<padded_size_per_row <<ENDL();

    const uint32_t ncores = get_compile_time_arg_val(8);
    // DPRINT << "ncores: " <<ncores <<ENDL();

    const uint32_t third_dim = get_compile_time_arg_val(9);
    // DPRINT << "third_dim: " <<third_dim <<ENDL();

    const uint32_t size_of_2d = get_compile_time_arg_val(10);
    // DPRINT<< "size_of_2d: " << size_of_2d << ENDL();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unpadded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(2);
    const uint32_t pad_value = get_arg_val<uint32_t>(3);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(4);  // this is passed and needs to change
    const uint32_t n_block_reps = get_arg_val<uint32_t>(5);
    const uint32_t core_number = get_arg_val<uint32_t>(6);

    /*
    DPRINT << "src addr: " << src_addr <<ENDL();
    DPRINT << "unpadded_X_size: " << unpadded_X_size <<ENDL();
    DPRINT << "padded_X_size: " << padded_X_size <<ENDL();
    DPRINT << "pad_value: " << pad_value <<ENDL();
    DPRINT << "start_stick_id: " << start_stick_id <<ENDL();
    DPRINT << "n_block_reps: " << n_block_reps <<ENDL();
    DPRINT << "core_number: " << core_number <<ENDL();
    */

    const uint32_t num_tiles_per_row =
        padded_X_size >> tile_row_shift_bits;  // means / 64, assuming bfloat16, there are 64 bytes per tile row
    // DPRINT << "num_tiles_per_row: " << num_tiles_per_row <<ENDL();

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    // DPRINT << "src0_is_dram: " << (uint32_t)src0_is_dram <<ENDL();

#define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
#if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<src0_is_dram> s = {
        .bank_base_address = src_addr, .log_base_2_of_page_size = log_base_2_of_page_size};
#else
    const InterleavedAddrGen<src0_is_dram> s = {.bank_base_address = src_addr, .page_size = unpadded_X_size};
#endif

    auto pad_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_reserve_back(cb_id_in0, num_tiles_per_row);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            // pad the tile by reading values from zero buffer in L1
            fill_with_val(l1_write_addr, padded_X_size << 3, pad_value);
            cb_push_back(cb_id_in0, num_tiles_per_row);
        }
    };
    auto read_block = [&](uint32_t num_rows,
                          uint32_t mul,
                          uint32_t size_per_row_per_block,
                          uint32_t padded_size_per_row,
                          uint32_t start_id,
                          uint32_t width_size,
                          uint32_t size_2d) {
        /*
        DPRINT << "read blck INPUTS " << ENDL();
        DPRINT << "num_rows: " << num_rows << ENDL();
        DPRINT << "mul: " << mul << ENDL();
        DPRINT << "size_per_row_per_block: " << size_per_row_per_block << ENDL();
        DPRINT << "padded_size_per_row: " << padded_size_per_row << ENDL();
        DPRINT << "start_id: " << start_id << ENDL();
        DPRINT << "width_size: " << width_size << ENDL();
        */

        uint32_t padding_rows = num_padding_rows & 31;  // 29;//(tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;
        // DPRINT << "padding_rows: " << padding_rows << ENDL();
        // DPRINT << "has rows: " << (uint32_t)has_rows <<ENDL();

        // DPRINT << "we are reserving in cb :" << (uint32_t) (tiles_per_col * has_rows) << ENDL();
        cb_reserve_back(cb_id_in0, 1 * has_rows);  // 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint32_t original_addr = get_write_ptr(cb_id_in0);
        // DPRINT << "l1_write_addr " << l1_write_addr << ENDL();
        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t src_noc_addr = get_noc_addr(size_2d + k, s);
            // DPRINT << "get address at: " <<  size_2d + k <<" which is " << (uint32_t)src_noc_addr <<ENDL();

            // Read from DRAM to tmp buffer
            noc_async_read(
                src_noc_addr + start_id + mul * size_per_row_per_block,
                l1_write_addr,
                width_size);  // was copying size per row and without + start_id // was padded_size_pwe_row instead of
                              // size_per_row_per_block

            // DPRINT <<"after reading we are at addr " << l1_write_addr + width_size << ENDL();
            uint32_t prev_size = mul * size_per_row_per_block + start_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            if (mul == ncores - 1 && this_block_size < width_size) {
                uint32_t to_pad = width_size - this_block_size;
                // DPRINT << "Padding start at "<< l1_write_addr + this_block_size << ENDL();
                fill_with_val(
                    l1_write_addr + this_block_size,
                    (to_pad) >> 2,
                    pad_value);  // padding next column for 32 values not remaining of row
            } else if (mul == ncores - 1 && prev_size > unpadded_X_size) {
                // DPRINT << "Padding IN ELSE STARTS AT "<< l1_write_addr << ENDL();
                fill_with_val(l1_write_addr, (width_size) >> 2, pad_value);
            }

            /*
            uint32_t prev_size = mul * size_per_row_per_block + start_id;
            uint32_t this_block_size = unpadded_X_size - prev_size;
            uint32_t to_pad = padded_X_size - this_block_size > width_size ? width_size: padded_X_size -
            this_block_size; if (mul == ncores - 1 && to_pad > 0) { fill_with_val(l1_write_addr + this_block_size,
            (to_pad) >> 2, pad_value); //padding next column for 32 values not remaining of row
            }
            */

            // Block before copying data from tmp to cb buffer
            noc_async_read_barrier();
            l1_write_addr += width_size;
            // DPRINT << "after finishing tihs row, addr is "<< l1_write_addr <<ENDL();
            if (k > 0 && k % 32 == 0) {
                cb_push_back(cb_id_in0, 1 * has_rows);
                cb_reserve_back(cb_id_in0, 1 * has_rows);  // 1);
            }
        }

        fill_with_val(l1_write_addr, padding_rows * (width_size >> 2), pad_value);  // width size was padded_X_size
        l1_write_addr += padding_rows * width_size;
        /*
        uint32_t prev_size2 = mul * size_per_row_per_block + start_id;
        uint32_t this_block_size2 = unpadded_X_size - prev_size2;
        if (mul == ncores - 1 && this_block_size2 < width_size) {
            uint32_t to_pad2 = width_size - this_block_size2;
            DPRINT << "Padding start at "<< l1_write_addr + this_block_size2 << ENDL();
            fill_with_val(l1_write_addr + this_block_size2, padding_rows * (to_pad2) >> 2, pad_value); //padding next
        column for 32 values not remaining of row
        }
        else if (mul == ncores - 1 && prev_size2 > unpadded_X_size) {
            DPRINT << "Padding IN ELSE STARTS AT "<< l1_write_addr << ENDL();
            fill_with_val(l1_write_addr, padding_rows * (width_size) >> 2, pad_value);
        }
        */

        // auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
        // for (uint32_t ii1 = 0; ii1 < 2048; ii1 = ii1+1) {
        //     if (ii1 % 16 == 0) {
        //         DPRINT << "CHECK HERE ";
        //     }
        //     DPRINT << "value at i1 = " << (uint32_t)ii1 <<  " is: " << BF16((uint16_t)ptr_orig[ii1]) << ENDL();
        // }

        cb_push_back(cb_id_in0, 1 * has_rows);
    };

    uint32_t stick_id = start_stick_id;
    uint32_t rt_arg_idx = 9;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    const uint32_t size_per_row_per_block = get_arg_val<uint32_t>(7);
    // DPRINT<< "size_per_row_per_block: " << size_per_row_per_block << ENDL();
    const uint32_t blocks_per_core = get_arg_val<uint32_t>(8);
    // DPRINT<< "blocks_per_core: " << blocks_per_core << ENDL();
    const uint32_t width_size = get_arg_val<uint32_t>(9);
    // DPRINT<< "width_size: " << width_size << ENDL();

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_id = 0;
        for (uint32_t b = 0; b < blocks_per_core; b++) {
            // DPRINT <<" BLOCK NUMBER " << b <<ENDL();
            read_block(
                total_num_rows,
                core_number,
                size_per_row_per_block,
                padded_size_per_row,
                start_id,
                width_size,
                size_2d);
            start_id += width_size;
        }
        size_2d += total_num_rows;
    }

    /*
    for (uint32_t block_rep_idx = 0; block_rep_idx < 1; ++block_rep_idx) {
        const uint32_t size_per_row = get_arg_val<uint32_t>(7);
        DPRINT<< "size_per_row: " << size_per_row << ENDL();
        DPRINT << "this is for block " << block_rep_idx << ENDL();
        const uint32_t repeat_count =
            get_arg_val<uint32_t>(rt_arg_idx + repeat_ct_idx);  // number of times the same block representation is used
        const uint32_t n_data = get_arg_val<uint32_t>(rt_arg_idx);  // number of full tile-rows
        const uint32_t n_mixed =
            get_arg_val<uint32_t>(rt_arg_idx + n_mixed_idx);  // number of rows in a partially filled tile-row
        const uint32_t n_pads = get_arg_val<uint32_t>(rt_arg_idx + n_pad_idx);  // number of padding tile-rows
        const uint32_t times =
            get_arg_val<uint32_t>(rt_arg_idx + times_idx);  // number of times the pattern of tile-rows repeats
        DPRINT << "n_data: " << n_data <<ENDL();
        DPRINT << "n_mixed: " << n_mixed <<ENDL();
        DPRINT << "n_pads: " << n_pads <<ENDL();
        DPRINT << "times: " << times <<ENDL();
        if (count == repeat_count) {
            rt_arg_idx = rt_arg_idx + num_rt_idx;
            count = 1;
        } else {
            count++;
        }
        for (uint32_t t = 0; t < times; ++t) {
            DPRINT << "Looping times time" <<ENDL();
            for (uint32_t y_t = 0; y_t < n_data; y_t++) {
                DPRINT << "Looping n_data times" <<ENDL();

                stick_id += tile_height;
            }

            read_block(total_num_rows, core_number, size_per_row, padded_size_per_row);
            stick_id += n_mixed;

            pad_blocks(n_pads);
        }
    }
    */
}
