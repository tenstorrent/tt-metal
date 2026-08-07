// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t bytes_per_tile_row = get_compile_time_arg_val(0);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    // Constexpr
    constexpr uint32_t dfb_id_in0 = 0;
    constexpr uint32_t tile_height = 32;

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_unpadded_W = get_arg_val<uint32_t>(1);
    const uint32_t padded_W_diff_blocks = get_arg_val<uint32_t>(2);
    const uint32_t num_unpadded_Z = get_arg_val<uint32_t>(3);
    const uint32_t padded_Z_diff_blocks = get_arg_val<uint32_t>(4);
    const uint32_t num_unpadded_Y = get_arg_val<uint32_t>(5);
    const uint32_t padded_Y_diff_blocks = get_arg_val<uint32_t>(6);
    const uint32_t num_leftover_Y = get_arg_val<uint32_t>(7);
    const uint32_t num_unpadded_X = get_arg_val<uint32_t>(8);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(9);
    const uint32_t pad_value = get_arg_val<uint32_t>(10);
    const uint32_t num_blocks_w_input = get_arg_val<uint32_t>(11);
    const uint32_t num_blocks_w_output = get_arg_val<uint32_t>(12);
    const uint32_t num_blocks_w_diff = get_arg_val<uint32_t>(13);
    const uint32_t block_row_size = get_arg_val<uint32_t>(14);
    const uint32_t block_row_leftover_size = get_arg_val<uint32_t>(15);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_block_c =
        block_row_size / bytes_per_tile_row;  // Assuming 2 bytes per datum, there are 64 bytes per tile row

    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

    uint32_t stick_id = 0;

    auto pad_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            dfb_in0.reserve_back(num_tiles_block_c);
            uint32_t l1_write_addr = dfb_in0.get_write_ptr();
            const uint32_t pad_bytes = block_row_size * 32;  // 32 = tile_height * 4 bytes/u32
            if (pad_value == 0) {
                // Fast path: zero-fill via NOC read from the zero source instead of scalar stores.
                noc.async_write_zeros(dfb_in0, pad_bytes);
                noc.write_zeros_l1_barrier();
            } else {
                volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
                // 8 = tile_height / 4
                for (uint32_t z = 0; z < block_row_size * 8; z++) {
                    dst[z] = pad_value;
                }
            }
            dfb_in0.push_back(num_tiles_block_c);
        }
    };

    auto read_block = [&](uint32_t base_stick_id, uint32_t num_rows, uint32_t offset, uint32_t block_size) {
        dfb_in0.reserve_back(num_tiles_block_c);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        const uint32_t entry_base = l1_write_addr;
        uint32_t curr_stick_id = base_stick_id;
        for (uint32_t k = 0; k < num_rows; k++) {
            CoreLocalMem<uint32_t> dst_mem(l1_write_addr);
            noc.async_read(
                s, dst_mem, block_size, {.page_id = curr_stick_id + k, .offset_bytes = offset}, {.offset_bytes = 0});

            if (block_row_size > block_size) {
                const uint32_t tail_bytes = block_row_size - block_size;
                if (pad_value == 0) {
                    noc.async_write_zeros(
                        dfb_in0, tail_bytes, {.offset_bytes = l1_write_addr + block_size - entry_base});
                } else {
                    dataflow_kernel_lib::fill_l1_range<elem_size>(l1_write_addr + block_size, tail_bytes, pad_value);
                }
            }

            // Block before copying data from tmp to cb buffer (and any zero-fill NOC reads issued above)
            noc.async_read_barrier();
            l1_write_addr += block_row_size;
        }
        if (num_rows < tile_height) {
            const uint32_t leftover_bytes = block_row_size * (tile_height - num_rows);
            if (pad_value == 0) {
                noc.async_write_zeros(dfb_in0, leftover_bytes, {.offset_bytes = l1_write_addr - entry_base});
                noc.write_zeros_l1_barrier();
            } else {
                volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
                for (uint32_t z = 0; z < leftover_bytes / 4; z++) {
                    dst[z] = pad_value;
                }
            }
        }
        dfb_in0.push_back(num_tiles_block_c);
    };

    auto read_block_rows = [&](uint32_t base_stick_id, uint32_t num_rows_block) {
        uint32_t block_row_offset = 0;

        for (uint32_t block_w = 0; block_w < num_blocks_w_input; block_w++) {
            read_block(base_stick_id, num_rows_block, block_row_offset, block_row_size);
            block_row_offset += block_row_size;
        }

        if (block_row_leftover_size > 0) {
            read_block(base_stick_id, num_rows_block, block_row_offset, block_row_leftover_size);
            block_row_offset += block_row_size;
        }
    };

    for (uint32_t w = 0; w < num_unpadded_W; w++) {
        for (uint32_t z = 0; z < num_unpadded_Z; z++) {
            for (uint32_t y_t = 0; y_t < num_unpadded_Y / tile_height; y_t++) {
                read_block_rows(stick_id, tile_height);
                // Read fully padded blocks
                pad_blocks(num_blocks_w_diff);
                stick_id += tile_height;
            }

            if (num_leftover_Y > 0) {
                read_block_rows(stick_id, num_leftover_Y);
                // Read fully padded blocks
                pad_blocks(num_blocks_w_diff);
                stick_id += num_leftover_Y;
            }
            pad_blocks(padded_Y_diff_blocks);
        }
        pad_blocks(padded_Z_diff_blocks);
    }
    pad_blocks(padded_W_diff_blocks);
}
