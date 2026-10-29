// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t num_rows = get_arg(args::num_rows);
    uint32_t half_row_width = get_arg(args::half_row_width);
    uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_out_no_mul(dfb::out_no_mul);
    DataflowBuffer dfb_out_mul(dfb::out_mul);

    // ublocks size defined in tiles
    const uint32_t tile_bytes = dfb_out_no_mul.get_tile_size();

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::dst);

    uint32_t out_no_mul_curr_id = start_id + half_row_width;
    uint32_t out_mul_curr_id = start_id;
    // read a ublock of tiles from src to DFB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < half_row_width; j++) {
            dfb_out_no_mul.wait_front(onetile);
            uint32_t out_no_mul_l1_read_addr = dfb_out_no_mul.get_read_ptr();
            noc.async_write(
                CoreLocalMem<uint32_t>(out_no_mul_l1_read_addr), s, tile_bytes, {}, {.page_id = out_no_mul_curr_id});
            noc.async_write_barrier();
            dfb_out_no_mul.pop_front(onetile);
            out_no_mul_curr_id++;

            dfb_out_mul.wait_front(onetile);
            uint32_t out_mul_l1_read_addr = dfb_out_mul.get_read_ptr();
            noc.async_write(
                CoreLocalMem<uint32_t>(out_mul_l1_read_addr), s, tile_bytes, {}, {.page_id = out_mul_curr_id});
            noc.async_write_barrier();
            dfb_out_mul.pop_front(onetile);
            out_mul_curr_id++;
        }
        out_no_mul_curr_id += half_row_width;
        out_mul_curr_id += half_row_width;
    }
}
