// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    constexpr uint32_t dfb_id_in0 = 0;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t tile_row_shift_bits = get_compile_time_arg_val(0);
    constexpr uint32_t unpadded_X_size = get_compile_time_arg_val(1);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_pages_in_row = get_compile_time_arg_val(3);
    constexpr uint32_t page_size = get_compile_time_arg_val(4);
    constexpr uint32_t size_of_valid_data_in_last_page_in_row = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_X_size = get_arg_val<uint32_t>(1);
    const uint32_t pad_value = get_arg_val<uint32_t>(2);
    const uint32_t start_page_id = get_arg_val<uint32_t>(3);
    const uint32_t n_block_reps = get_arg_val<uint32_t>(4);

    const uint32_t num_tiles_per_row =
        padded_X_size >> tile_row_shift_bits;  // means / 64, assuming bfloat16, there are 64 bytes per tile row

    const auto s = TensorAccessor(src_args, src_addr);
    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);

    auto pad_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            dfb_in0.reserve_back(num_tiles_per_row);
            uint32_t l1_write_addr = dfb_in0.get_write_ptr();
            // pad the tile by reading values from zero buffer in L1
            dataflow_kernel_lib::fill_l1_range<elem_size>(
                l1_write_addr, padded_X_size << 5, pad_value);  // "<< 5" = "* tile_height"
            dfb_in0.push_back(num_tiles_per_row);
        }
    };

    auto read_block = [&](uint32_t base_page_id, uint32_t num_rows) {
        uint32_t padding_rows = (tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        dfb_in0.reserve_back(num_tiles_per_row * has_rows);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t k = 0; k < num_rows; k++) {
            uint32_t start_of_row_l1_write_addr = l1_write_addr;
            for (uint32_t i = 0; i < num_pages_in_row - 1; i++) {
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    s,
                    dst,
                    page_size,
                    {.page_id = base_page_id + k * num_pages_in_row + i, .offset_bytes = 0},
                    {.offset_bytes = 0});
                l1_write_addr += page_size;
            }
            // Process the last page in a row separately, as it may have padding at the end
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                s,
                dst,
                size_of_valid_data_in_last_page_in_row,
                {.page_id = base_page_id + k * num_pages_in_row + num_pages_in_row - 1, .offset_bytes = 0},
                {.offset_bytes = 0});
            uint32_t size_of_padding_columns = padded_X_size - unpadded_X_size;
            dataflow_kernel_lib::fill_l1_range<elem_size>(
                start_of_row_l1_write_addr + unpadded_X_size, size_of_padding_columns, pad_value);
            l1_write_addr += size_of_valid_data_in_last_page_in_row + size_of_padding_columns;
        }

        dataflow_kernel_lib::fill_l1_range<elem_size>(l1_write_addr, padding_rows * padded_X_size, pad_value);
        noc.async_read_barrier();
        dfb_in0.push_back(num_tiles_per_row * has_rows);
    };

    uint32_t page_id = start_page_id;
    uint32_t rt_arg_idx = 5;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
        const uint32_t repeat_count =
            get_arg_val<uint32_t>(rt_arg_idx + repeat_ct_idx);  // number of times the same block representation is used
        const uint32_t n_data = get_arg_val<uint32_t>(rt_arg_idx);  // number of full tile-rows
        const uint32_t n_mixed =
            get_arg_val<uint32_t>(rt_arg_idx + n_mixed_idx);  // number of rows in a partially filled tile-row
        const uint32_t n_pads = get_arg_val<uint32_t>(rt_arg_idx + n_pad_idx);  // number of padding tile-rows
        const uint32_t times =
            get_arg_val<uint32_t>(rt_arg_idx + times_idx);  // number of times the pattern of tile-rows repeats
        if (count == repeat_count) {
            rt_arg_idx = rt_arg_idx + num_rt_idx;
            count = 1;
        } else {
            count++;
        }
        for (uint32_t t = 0; t < times; ++t) {
            for (uint32_t y_t = 0; y_t < n_data; y_t++) {
                read_block(page_id, tile_height);
                page_id += tile_height * num_pages_in_row;
            }

            read_block(page_id, n_mixed);
            page_id += n_mixed * num_pages_in_row;

            pad_blocks(n_pads);
        }
    }
}
