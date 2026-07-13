// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t total_num_rows = get_arg(args::total_num_rows);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t unpadded_X_size = get_arg(args::unpadded_X_size);

    const auto s = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer cb_out0(dfb::out);

    auto write_block = [&](uint32_t num_rows,
                           uint32_t start_row_id,
                           uint32_t start_column_id,
                           uint32_t width_size,
                           uint32_t size_2d,
                           uint32_t single_block_size) {
        bool has_rows = (num_rows) > 0;

        cb_out0.wait_front(single_block_size * has_rows);
        // Read from the front of the waited entries. The legacy kernel used get_write_ptr() here,
        // which only coincides with the front when the CB is sized EXACTLY to one block (full ->
        // write ptr wraps to read ptr). The Metal 2.0 port shares one DFB sized to the max region
        // block count across sub-regions, so smaller regions never fill it; get_read_ptr() is the
        // size-independent address of the block's data.
        uint32_t l1_read_addr = cb_out0.get_read_ptr();

        for (uint32_t k = start_row_id; k < start_row_id + num_rows; k++) {
            uint32_t total_size = start_column_id + width_size;
            uint32_t write_size = width_size;

            if (total_size > unpadded_X_size) {
                uint32_t padded_size = total_size - unpadded_X_size;
                write_size -= padded_size;
            }

            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src, s, write_size, {.offset_bytes = 0}, {.page_id = size_2d + k, .offset_bytes = start_column_id});

            noc.async_write_barrier();

            l1_read_addr += width_size;
        }

        cb_out0.pop_front(single_block_size * has_rows);
    };

    const uint32_t width_size = get_arg(args::width_size);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_row_id = get_arg(args::start_row_id);
        uint32_t start_column_id = get_arg(args::start_column_id);
        uint32_t single_block_size_row_arg = get_arg(args::single_block_size_row_arg);
        uint32_t single_block_size_col_arg = get_arg(args::single_block_size_col_arg);
        uint32_t sub_block_width_size = get_arg(args::sub_block_width_size);
        uint32_t single_sub_block_size_row_arg = get_arg(args::single_sub_block_size_row_arg);

        for (uint32_t b = 0; b < single_block_size_col_arg; b++) {
            uint32_t this_block_num_rows = tile_height;
            if (start_row_id + tile_height > total_num_rows) {
                this_block_num_rows = total_num_rows - start_row_id;
            }
            for (uint32_t m = 0; m < width_size; m += sub_block_width_size) {
                uint32_t start_column_id_u = start_column_id + m;
                if (this_block_num_rows > 0) {
                    write_block(
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
