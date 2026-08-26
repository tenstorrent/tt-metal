// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t half_row_width = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out_no_mul = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out_mul = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    // ublocks size defined in tiles
    const uint32_t tile_bytes = get_tile_size(cb_id_out_no_mul);

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(dst_args, dst_addr);

    CircularBuffer cb_out_no_mul(cb_id_out_no_mul);
    CircularBuffer cb_out_mul(cb_id_out_mul);

    uint32_t out_no_mul_curr_id = start_id + half_row_width;
    uint32_t out_mul_curr_id = start_id;
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < half_row_width; j++) {
            cb_out_no_mul.wait_front(onetile);
            uint32_t out_no_mul_l1_read_addr = cb_out_no_mul.get_read_ptr();
            noc.async_write(
                CoreLocalMem<uint32_t>(out_no_mul_l1_read_addr), s, tile_bytes, {}, {.page_id = out_no_mul_curr_id});
            noc.async_write_barrier();
            cb_out_no_mul.pop_front(onetile);
            out_no_mul_curr_id++;

            cb_out_mul.wait_front(onetile);
            uint32_t out_mul_l1_read_addr = cb_out_mul.get_read_ptr();
            noc.async_write(
                CoreLocalMem<uint32_t>(out_mul_l1_read_addr), s, tile_bytes, {}, {.page_id = out_mul_curr_id});
            noc.async_write_barrier();
            cb_out_mul.pop_front(onetile);
            out_mul_curr_id++;
        }
        out_no_mul_curr_id += half_row_width;
        out_mul_curr_id += half_row_width;
    }
}
