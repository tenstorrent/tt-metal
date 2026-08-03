// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Handles unaligned head/tail with element-sized stores, then writes 4B words in the aligned middle.
// Assumption: for val_size < 4, 'val' must have the element already replicated into all sub-word
// positions (e.g. two bfloat16 values packed into a uint32_t).
template <uint32_t val_size>
FORCE_INLINE void fill_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(val_size == 1 || val_size == 2 || val_size == 4, "Unsupported val_size");
    using IntType =
        std::conditional_t<(val_size == 1), uint8_t, std::conditional_t<(val_size == 2), uint16_t, uint32_t>>;

    uint32_t end_addr = start_addr + n_bytes;
    uint32_t start_addr_4B = (start_addr + 0x3) & 0xFFFFFFFC;
    uint32_t end_addr_4B = end_addr & 0xFFFFFFFC;

    auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
    auto* end_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
    for (auto* ptr = start_ptr_4B; ptr < end_ptr_4B; ++ptr) {
        *ptr = val;
    }

    if constexpr (val_size < 4) {
        auto* start_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr);
        auto* end_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr);
        auto* start_ptr_4B_e = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr_4B);
        auto* end_ptr_4B_e = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr_4B);
        IntType val_ = static_cast<IntType>(val);
        for (auto* ptr = start_ptr; ptr < start_ptr_4B_e; ++ptr) {
            *ptr = val_;
        }
        for (auto* ptr = end_ptr_4B_e; ptr < end_ptr; ++ptr) {
            *ptr = val_;
        }
    }
}

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
            // pad the tile by reading values from zero buffer in L1
            volatile tt_l1_ptr std::uint32_t* dst = (volatile tt_l1_ptr uint32_t*)(l1_write_addr);
            // 8 = tile_height / 4
            for (uint32_t z = 0; z < block_row_size * 8; z++) {
                dst[z] = pad_value;
            }
            dfb_in0.push_back(num_tiles_block_c);
        }
    };

    auto read_block = [&](uint32_t base_stick_id, uint32_t num_rows, uint32_t offset, uint32_t block_size) {
        dfb_in0.reserve_back(num_tiles_block_c);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        uint32_t curr_stick_id = base_stick_id;
        for (uint32_t k = 0; k < num_rows; k++) {
            CoreLocalMem<uint32_t> dst_mem(l1_write_addr);
            noc.async_read(
                s, dst_mem, block_size, {.page_id = curr_stick_id + k, .offset_bytes = offset}, {.offset_bytes = 0});

            if (block_row_size > block_size) {
                fill_with_val<elem_size>(l1_write_addr + block_size, block_row_size - block_size, pad_value);
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
