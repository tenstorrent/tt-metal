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
    constexpr uint32_t ncores = get_arg(args::ncores);
    constexpr uint32_t third_dim = get_arg(args::third_dim);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    constexpr uint32_t unpadded_X_size = get_arg(args::unpadded_X_size);

    // The output base address is carried by the TensorAccessor binding; the legacy dst_addr runtime
    // arg is gone.
    const uint32_t core_number = get_arg(args::core_number);

    const auto s = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer cb_out0(dfb::out);

    auto write_block = [&](uint32_t num_rows,
                           uint32_t mul,
                           uint32_t size_per_row_per_block,
                           uint32_t start_id,
                           uint32_t width_size,
                           uint32_t size_2d) {
        uint32_t onetile = 1;
        bool has_rows = (num_rows) > 0;

        cb_out0.wait_front(onetile * has_rows);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();

        for (uint32_t k = 0; k < num_rows; k++) {
            uint32_t total_size = mul * size_per_row_per_block + start_id + width_size;
            uint32_t padded_size = total_size - unpadded_X_size;
            uint32_t write_size = width_size;

            if (mul == ncores - 1 && padded_size > 0) {
                write_size = width_size - padded_size;
            }

            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                write_size,
                {.offset_bytes = 0},
                {.page_id = size_2d + k, .offset_bytes = start_id + mul * size_per_row_per_block});

            noc.async_write_barrier();

            if (k > 0 && (k % tile_width == 0)) {
                cb_out0.pop_front(onetile * has_rows);
                cb_out0.wait_front(onetile * has_rows);
            }
            l1_read_addr += width_size;
        }

        cb_out0.pop_front(onetile * has_rows);
    };

    const uint32_t size_per_row_per_block = get_arg(args::size_per_row_per_block);
    const uint32_t blocks_per_core = get_arg(args::blocks_per_core);
    const uint32_t width_size = get_arg(args::width_size);

    uint32_t size_2d = 0;
    for (uint32_t dim3 = 0; dim3 < third_dim; dim3++) {
        uint32_t start_id = 0;
        for (uint32_t b = 0; b < blocks_per_core; b++) {
            write_block(total_num_rows, core_number, size_per_row_per_block, start_id, width_size, size_2d);
            start_id += width_size;
        }
        size_2d += total_num_rows;
    }
}
