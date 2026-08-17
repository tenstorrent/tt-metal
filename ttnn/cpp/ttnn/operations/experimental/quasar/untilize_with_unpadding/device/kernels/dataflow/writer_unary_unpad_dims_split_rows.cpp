// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

inline uint64_t round_down_32(uint64_t a) { return (a >> 5) << 5; }

void kernel_main() {
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
    const uint32_t num_blocks_w_input = get_arg(args::num_blocks_w_input);
    const uint32_t num_blocks_w_output = get_arg(args::num_blocks_w_output);
    const uint32_t num_blocks_w_diff = get_arg(args::num_blocks_w_diff);
    const uint32_t block_row_size = get_arg(args::block_row_size);
    const uint32_t block_row_leftover_size = get_arg(args::block_row_leftover_size);

    uint32_t stick_id = 0;

    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;

    const uint32_t num_tiles_block_c =
        FLOAT32_DTYPE ? block_row_size / 128
                      : block_row_size / 64;  // Assuming 4 / 2 bytes per datum, there are 128 / 64 bytes per tile row

    const auto s = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer cb_out0(dfb::out);

    auto pop_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_out0.wait_front(num_tiles_block_c);
            cb_out0.pop_front(num_tiles_block_c);
        }
    };

    auto write_block = [&](uint32_t base_stick_id, uint32_t num_rows, uint32_t offset, uint32_t block_size) {
        cb_out0.wait_front(num_tiles_block_c);
        uint32_t curr_stick_id = base_stick_id;
        for (uint32_t k = 0; k < num_rows; k++) {
            noc.async_write(
                cb_out0,
                s,
                block_size,
                {.offset_bytes = k * block_row_size},
                {.page_id = curr_stick_id, .offset_bytes = offset});

            curr_stick_id++;

            // Block write
            noc.async_write_barrier();
        }
        cb_out0.pop_front(num_tiles_block_c);
    };

    auto write_block_rows = [&](uint32_t num_rows_block, uint32_t base_stick_id) {
        uint32_t block_row_offset = 0;
        for (uint32_t block_w = 0; block_w < num_blocks_w_output; block_w++) {
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
