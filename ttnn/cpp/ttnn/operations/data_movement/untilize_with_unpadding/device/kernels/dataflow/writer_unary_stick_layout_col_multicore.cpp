// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr std::uint32_t dfb_id_out0 = 16;

    constexpr std::uint32_t total_num_rows = get_compile_time_arg_val(0);
    constexpr std::uint32_t ncores = get_compile_time_arg_val(1);
    constexpr std::uint32_t third_dim = get_compile_time_arg_val(2);
    constexpr std::uint32_t tile_width = get_compile_time_arg_val(3);
    constexpr std::uint32_t unpadded_X_size = get_compile_time_arg_val(4);
    constexpr auto dst_args = TensorAccessorArgs<5>();

    const std::uint32_t dst_addr = get_arg_val<std::uint32_t>(0);
    const std::uint32_t core_number = get_arg_val<std::uint32_t>(1);

    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out0(dfb_id_out0);

    auto write_block = [&](std::uint32_t num_rows,
                           std::uint32_t mul,
                           std::uint32_t size_per_row_per_block,
                           std::uint32_t start_id,
                           std::uint32_t width_size,
                           std::uint32_t size_2d) {
        std::uint32_t onetile = 1;
        bool has_rows = (num_rows) > 0;

        dfb_out0.wait_front(onetile * has_rows);
        std::uint32_t l1_read_addr = dfb_out0.get_read_ptr();

        for (std::uint32_t k = 0; k < num_rows; k++) {
            std::uint32_t total_size = mul * size_per_row_per_block + start_id + width_size;
            std::uint32_t padded_size = total_size - unpadded_X_size;
            std::uint32_t write_size = width_size;

            if (mul == ncores - 1 && padded_size > 0) {
                write_size = width_size - padded_size;
            }

            CoreLocalMem<std::uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                write_size,
                {.offset_bytes = 0},
                {.page_id = size_2d + k, .offset_bytes = start_id + mul * size_per_row_per_block});

            noc.async_write_barrier();

            if (k > 0 && (k % tile_width == 0)) {
                dfb_out0.pop_front(onetile * has_rows);
                dfb_out0.wait_front(onetile * has_rows);
            }
            l1_read_addr += width_size;
        }

        dfb_out0.pop_front(onetile * has_rows);
    };

    const std::uint32_t size_per_row_per_block = get_arg_val<std::uint32_t>(3);
    const std::uint32_t blocks_per_core = get_arg_val<std::uint32_t>(4);
    const std::uint32_t width_size = get_arg_val<std::uint32_t>(5);

    std::uint32_t size_2d = 0;
    for (std::uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        std::uint32_t start_id = 0;
        for (std::uint32_t b = 0; b < blocks_per_core; b++) {
            write_block(total_num_rows, core_number, size_per_row_per_block, start_id, width_size, size_2d);
            start_id += width_size;
        }
        size_2d += total_num_rows;
    }
}
