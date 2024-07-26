// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

inline uint64_t round_down_32(uint64_t a){
    return (a >> 5) << 5;
}

void kernel_main() {

    // Constexpr
    constexpr uint32_t cb_id_out0                      = 16;
    constexpr uint32_t tile_height                     = 32;

    const uint32_t dst_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t num_unpadded_W           = get_arg_val<uint32_t>(1);
    const uint32_t padded_W_diff_blocks     = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_Z           = get_arg_val<uint32_t>(3);
    const uint32_t padded_Z_diff_blocks     = get_arg_val<uint32_t>(4);
    const uint32_t num_unpadded_Y           = get_arg_val<uint32_t>(5);
    const uint32_t padded_Y_diff_blocks     = get_arg_val<uint32_t>(6);
    const uint32_t num_leftover_Y           = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_X           = get_arg_val<uint32_t>(8);
    const uint32_t unpadded_X_size          = get_arg_val<uint32_t>(9);
    const uint32_t padded_X_size            = get_arg_val<uint32_t>(10);
    const uint32_t num_blocks_w_input       = get_arg_val<uint32_t>(11);
    const uint32_t num_blocks_w_output      = get_arg_val<uint32_t>(12);
    const uint32_t num_blocks_w_diff        = get_arg_val<uint32_t>(13);
    const uint32_t block_row_size           = get_arg_val<uint32_t>(14);
    const uint32_t block_row_leftover_size  = get_arg_val<uint32_t>(15);


    uint32_t stick_id          = 0;

    constexpr bool dst_is_dram          = get_compile_time_arg_val(0) == 1;
    #define stick_size_is_pow2 get_compile_time_arg_val(1) == 1
    constexpr bool FLOAT32_DTYPE          = get_compile_time_arg_val(3) == 1;

    const uint32_t num_tiles_block_c = FLOAT32_DTYPE ? block_row_size / 128 : block_row_size / 64; // Assuming 4 / 2 bytes per datum, there are 128 / 64 bytes per tile row

    #if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(2);
    const InterleavedPow2AddrGen<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size
    };
    #else
    const InterleavedAddrGen<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = unpadded_X_size
    };
    #endif

    auto pop_blocks = [&] (uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_wait_front(cb_id_out0, num_tiles_block_c);
            cb_pop_front(cb_id_out0, num_tiles_block_c);
        }
    };

    auto write_block = [&] (uint32_t base_stick_id, uint32_t num_rows, uint32_t offset, uint32_t block_size) {
        cb_wait_front(cb_id_out0, num_tiles_block_c);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        uint32_t curr_stick_id = base_stick_id;
        for (uint32_t k = 0; k < num_rows; k++) {
            uint64_t dst_noc_addr = get_noc_addr(curr_stick_id, s) + offset;

            // Write out tmp buffer
            noc_async_write(l1_read_addr, dst_noc_addr, block_size);

            l1_read_addr += block_row_size;
            curr_stick_id++;

            // Block write
            noc_async_write_barrier();
        }
        cb_pop_front(cb_id_out0, num_tiles_block_c);
    };

    auto write_block_rows = [&] (uint32_t num_rows_block, uint32_t base_stick_id) {
        uint32_t block_row_offset = 0;
        for(uint32_t block_w = 0; block_w < num_blocks_w_output; block_w++) {
            write_block(base_stick_id, num_rows_block, block_row_offset, block_row_size);
            block_row_offset += block_row_size;
        }

        // Change to define
        if (block_row_leftover_size > 0) {
            write_block(base_stick_id, num_rows_block, block_row_offset, block_row_leftover_size);
            block_row_offset += block_row_leftover_size;
        }
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t y_t = 0; y_t < num_unpadded_Y / tile_height; y_t++) {
                write_block_rows(tile_height, stick_id);
                pop_blocks(num_blocks_w_diff);
                stick_id += tile_height;
            }

            // Change to define
            if (num_leftover_Y > 0) {
                write_block_rows(num_leftover_Y, stick_id);
                pop_blocks(num_blocks_w_diff);
                stick_id += num_leftover_Y;

            }
            pop_blocks(padded_Y_diff_blocks);
        }
        pop_blocks(padded_Z_diff_blocks);
    }
    pop_blocks(padded_W_diff_blocks);

}
