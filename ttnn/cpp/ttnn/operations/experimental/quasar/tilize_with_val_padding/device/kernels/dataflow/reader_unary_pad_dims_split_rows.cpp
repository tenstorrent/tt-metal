// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t bytes_per_tile_row = get_arg(args::bytes_per_tile_row);

    // Constexpr
    constexpr uint32_t tile_height = 32;

    const uint32_t num_unpadded_W = get_arg(args::num_unpadded_W);
    const uint32_t padded_W_diff_blocks = get_arg(args::padded_W_diff_blocks);
    const uint32_t num_unpadded_Z = get_arg(args::num_unpadded_Z);
    const uint32_t padded_Z_diff_blocks = get_arg(args::padded_Z_diff_blocks);
    const uint32_t num_unpadded_Y = get_arg(args::num_unpadded_Y);
    const uint32_t padded_Y_diff_blocks = get_arg(args::padded_Y_diff_blocks);
    const uint32_t num_leftover_Y = get_arg(args::num_leftover_Y);
    const uint32_t num_unpadded_X = get_arg(args::num_unpadded_X);
    const uint32_t padded_X_size = get_arg(args::padded_X_size);
    const uint32_t pad_value = get_arg(args::pad_value);
    const uint32_t num_blocks_w_input = get_arg(args::num_blocks_w_input);
    const uint32_t num_blocks_w_output = get_arg(args::num_blocks_w_output);
    const uint32_t num_blocks_w_diff = get_arg(args::num_blocks_w_diff);
    const uint32_t block_row_size = get_arg(args::block_row_size);
    const uint32_t block_row_leftover_size = get_arg(args::block_row_leftover_size);

    // TODO(agrebenisan): This isn't good... here we are assuming
    // that the stick size dictates tiles c, but stick size
    // doesn't necessarily need to be divisible by tiles c...
    // this is only the case really for tilize
    const uint32_t num_tiles_block_c =
        block_row_size / bytes_per_tile_row;  // Assuming 2 bytes per datum, there are 64 bytes per tile row

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in);

    uint32_t stick_id = 0;

    auto pad_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_in0.reserve_back(num_tiles_block_c);
            uint32_t l1_write_addr = cb_in0.get_write_ptr();
            // pad the tile by reading values from zero buffer in L1
            volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
            // 8 = tile_height / 4
            for (uint32_t z = 0; z < block_row_size * 8; z++) {
                dst[z] = pad_value;
            }
            cb_in0.push_back(num_tiles_block_c);
        }
    };

    auto read_block = [&](uint32_t base_stick_id, uint32_t num_rows, uint32_t offset, uint32_t block_size) {
        cb_in0.reserve_back(num_tiles_block_c);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        uint32_t curr_stick_id = base_stick_id;
        for (uint32_t k = 0; k < num_rows; k++) {
            noc.async_read(
                s,
                cb_in0,
                block_size,
                {.page_id = curr_stick_id + k, .offset_bytes = offset},
                {.offset_bytes = k * block_row_size});

            if (block_row_size > block_size) {
                volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr + block_size);
                for (uint32_t z = 0; z < (block_row_size - block_size) / 4; z++) {
                    dst[z] = pad_value;
                }
            }

            // Block before copying data from tmp to cb buffer
            noc.async_read_barrier();
            l1_write_addr += block_row_size;
        }
        if (num_rows < tile_height) {
            volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);

            for (uint32_t z = 0; z < (block_row_size) / 4 * (tile_height - num_rows); z++) {
                dst[z] = pad_value;
            }
        }
        cb_in0.push_back(num_tiles_block_c);
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
