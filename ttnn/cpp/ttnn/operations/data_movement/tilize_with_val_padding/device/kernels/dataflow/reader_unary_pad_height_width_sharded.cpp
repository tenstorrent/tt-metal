// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_input_rows = get_arg(args::num_input_rows);
    const uint32_t input_width_bytes = get_arg(args::input_width_bytes);
    const uint32_t input_block_size = get_arg(args::input_block_size);
    const uint32_t num_padded_tiles_per_batch = get_arg(args::num_padded_tiles_per_batch);
    const uint32_t num_padded_rows = get_arg(args::num_padded_rows);
    const uint32_t num_batches = get_arg(args::num_batches);
    const uint32_t packed_pad_value = get_arg(args::packed_pad_value);

    Noc noc;
    // dfb_in0 is the input shard itself (a DFB on borrowed memory) — read-only here.
    // dfb_in1 is the row-major staging DFB the compute kernel tilizes from.
    // dfb_pad holds one row of the pad value, reused for every padded row.
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_pad(dfb::pad);

    dfb_in0.reserve_back(num_input_rows);
    dfb_in1.reserve_back(num_padded_tiles_per_batch);
    dfb_pad.reserve_back(1);

    uint32_t read_addr = dfb_in0.get_read_ptr();
    uint32_t write_addr = dfb_in1.get_write_ptr();
    uint32_t pad_addr = dfb_pad.get_write_ptr();

    {
        CoreLocalMem<uint32_t> dst(write_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            input_block_size,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = read_addr},
            {.offset_bytes = 0});
    }
    read_addr += input_block_size;
    write_addr += input_block_size;
    volatile tt_l1_ptr std::uint32_t* pad = (volatile tt_l1_ptr uint32_t*)(pad_addr);
    for (uint32_t i = 0; i < input_width_bytes >> 2; ++i) {
        pad[i] = packed_pad_value;
    }
    for (uint32_t i = 0; i < num_padded_rows; ++i) {
        CoreLocalMem<uint32_t> dst(write_addr);
        noc.async_read(
            UnicastEndpoint{},
            dst,
            input_width_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = pad_addr},
            {.offset_bytes = 0});
        write_addr += input_width_bytes;
    }
    noc.async_read_barrier();
    dfb_in1.push_back(num_padded_tiles_per_batch);

    for (uint32_t b = 1; b < num_batches; ++b) {
        dfb_in1.reserve_back(num_padded_tiles_per_batch);
        write_addr = dfb_in1.get_write_ptr();
        {
            CoreLocalMem<uint32_t> dst(write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                input_block_size,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = read_addr},
                {.offset_bytes = 0});
        }
        read_addr += input_block_size;
        write_addr += input_block_size;
        for (uint32_t i = 0; i < num_padded_rows; ++i) {
            CoreLocalMem<uint32_t> dst(write_addr);
            noc.async_read(
                UnicastEndpoint{},
                dst,
                input_width_bytes,
                {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                 .addr = pad_addr},
                {.offset_bytes = 0});
            write_addr += input_width_bytes;
        }
        noc.async_read_barrier();
        dfb_in1.push_back(num_padded_tiles_per_batch);
    }
}
