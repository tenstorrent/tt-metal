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
    uint32_t half_row_size = get_arg(args::half_row_size);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint16_t scalar_value = get_arg(args::scalar_value);

    DataflowBuffer dfb_in_no_mul(dfb::in_no_mul);
    DataflowBuffer dfb_in_mul(dfb::in_mul);
    DataflowBuffer dfb_in_scalar(dfb::in_scalar);

    // in_no_mul, in_mul are from same tensor, so same sizes
    const uint32_t tile_bytes = dfb_in_no_mul.get_tile_size();

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::src);

    // Fill tile with zeros
    dfb_in_scalar.reserve_back(onetile);
    uint32_t l1_zeros_addr_in_scalar = dfb_in_scalar.get_write_ptr();
    volatile tt_l1_ptr uint16_t* scalar_buffer =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    dfb_in_scalar.push_back(onetile);

    uint32_t in_no_mul_curr_id = start_id;
    uint32_t in_mul_curr_id = start_id + half_row_size;
    // read a ublock of tiles from src to DFB, and then push the ublock to unpacker
    for (uint32_t i = 0; i < num_rows; i++) {
        for (uint32_t j = 0; j < half_row_size; j++) {
            dfb_in_no_mul.reserve_back(onetile);
            uint32_t in_no_mul_l1_write_addr = dfb_in_no_mul.get_write_ptr();
            noc.async_read(
                s, CoreLocalMem<uint32_t>(in_no_mul_l1_write_addr), tile_bytes, {.page_id = in_no_mul_curr_id}, {});
            noc.async_read_barrier();
            dfb_in_no_mul.push_back(onetile);
            in_no_mul_curr_id++;

            dfb_in_mul.reserve_back(onetile);
            uint32_t in1_l1_write_addr = dfb_in_mul.get_write_ptr();
            noc.async_read(s, CoreLocalMem<uint32_t>(in1_l1_write_addr), tile_bytes, {.page_id = in_mul_curr_id}, {});
            noc.async_read_barrier();
            dfb_in_mul.push_back(onetile);
            in_mul_curr_id++;
        }
        in_no_mul_curr_id += half_row_size;
        in_mul_curr_id += half_row_size;
    }
}
