// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr uint32_t dfb_id_in0 = 0;
    constexpr uint32_t dfb_id_in1 = 1;

    constexpr uint32_t total_num_rows = get_compile_time_arg_val(0);
    constexpr uint32_t third_dim = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t element_size = get_compile_time_arg_val(3);
    constexpr uint32_t unpadded_X_size = get_compile_time_arg_val(4);
    constexpr uint32_t dram_alignment = get_compile_time_arg_val(5);
    constexpr uint64_t dram_align_mask = ~static_cast<uint64_t>(dram_alignment - 1);
    constexpr uint64_t dram_align_offset = static_cast<uint64_t>(dram_alignment - 1);
    constexpr auto src_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pad_value = get_arg_val<uint32_t>(1);

    const auto s = TensorAccessor(src_args, src_addr);
    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer dfb_in1(dfb_id_in1);

    dfb_in1.reserve_back(1);
    uint32_t temp_addr_raw = dfb_in1.get_write_ptr();
    uint32_t temp_addr = (temp_addr_raw + dram_alignment - 1) & ~(dram_alignment - 1);
    dfb_in1.push_back(1);

    auto read_block = [&](uint32_t num_rows,
                          uint32_t start_row_id,
                          uint32_t start_column_id,
                          uint32_t width_size,
                          uint32_t size_2d,
                          uint32_t single_block_size) {
        uint32_t padding_rows = num_rows == 32 ? 0 : 32 - num_rows;
        bool has_rows = (num_rows + padding_rows) > 0;

        dfb_in0.reserve_back(single_block_size * has_rows);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();

        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint64_t src_noc_addr = s.get_noc_addr(size_2d + k);
            if (((src_noc_addr + (uint64_t)start_column_id) & dram_align_offset) ==
                ((uint64_t)l1_write_addr & dram_align_offset)) {
                // Read from DRAM to tmp buffer
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(
                    s, dst, width_size, {.page_id = size_2d + k, .offset_bytes = start_column_id}, {.offset_bytes = 0});

                // Block before copying data from tmp to cb buffer
                noc.async_read_barrier();

                uint32_t prev_size = start_column_id;
                uint32_t this_block_size = unpadded_X_size - prev_size;
                if (this_block_size < width_size) {
                    uint32_t to_pad = width_size - this_block_size;
                    dataflow_kernel_lib::fill_l1_range<element_size>(
                        l1_write_addr + this_block_size, to_pad, pad_value);
                }
            } else {
                // If there is a mis-alignment, we first load the data to a middle L1 cb, then copy to the final cb
                // buffer. The aligned-down source is a full NoC address, so it is supplied directly via
                // UnicastEndpoint (decomposed into x/y/local addr; NOC_XY_ADDR repacks it unchanged).
                const uint64_t aligned_src_noc_addr = (src_noc_addr + (uint64_t)start_column_id) & dram_align_mask;
                CoreLocalMem<uint32_t> temp_dst(temp_addr);
                noc.async_read(
                    UnicastEndpoint{},
                    temp_dst,
                    width_size + dram_alignment,
                    {.noc_x = (uint32_t)NOC_UNICAST_ADDR_X(aligned_src_noc_addr),
                     .noc_y = (uint32_t)NOC_UNICAST_ADDR_Y(aligned_src_noc_addr),
                     .addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(aligned_src_noc_addr)},
                    {.offset_bytes = 0});

                // Block before copying data from tmp to cb buffer
                noc.async_read_barrier();

                uint32_t prev_size = start_column_id;
                uint32_t this_block_size = unpadded_X_size - prev_size;
                if (this_block_size < width_size) {
                    uint32_t to_pad = width_size - this_block_size;
                    dataflow_kernel_lib::fill_l1_range<element_size>(
                        temp_addr + ((src_noc_addr + (uint64_t)start_column_id) & dram_align_offset) + this_block_size,
                        to_pad,
                        pad_value);
                }

                tt_memmove<false, false, true, 0>(
                    noc,
                    l1_write_addr,
                    temp_addr + ((src_noc_addr + (uint64_t)start_column_id) & dram_align_offset),
                    width_size);
            }

            l1_write_addr += width_size;
        }

        const uint32_t row_pad_bytes = padding_rows * width_size;
        dataflow_kernel_lib::fill_l1_range<element_size>(l1_write_addr, row_pad_bytes, pad_value);
        l1_write_addr += row_pad_bytes;

        dfb_in0.push_back(single_block_size * has_rows);
    };

    const uint32_t width_size = get_arg_val<uint32_t>(2);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_row_id = get_arg_val<uint32_t>(3);
        uint32_t start_column_id = get_arg_val<uint32_t>(4);
        uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(5);
        uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(6);
        uint32_t sub_block_width_size = get_arg_val<uint32_t>(7);
        uint32_t single_sub_block_size_row_arg = get_arg_val<uint32_t>(8);

        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            if (this_block_num_rows > 0) {
                for (uint32_t m = 0; m < width_size; m += sub_block_width_size) {
                    uint32_t start_column_id_u = start_column_id + m;
                    read_block(
                        this_block_num_rows,
                        start_row_id,
                        start_column_id_u,
                        sub_block_width_size,
                        size_2d,
                        single_sub_block_size_row_arg);
                }
            }
            start_row_id += tile_height;
        }
        size_2d += total_num_rows;
    }
}
