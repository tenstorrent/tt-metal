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
    // src_shard: sharded input borrowed onto the input buffer (legacy c_1).
    // in:        tilize input stream produced for compute (legacy c_0).
    // pad:       reader-local scratch holding one padding row (legacy c_2).
    DataflowBuffer cb_in0(dfb::src_shard);
    DataflowBuffer cb_in1(dfb::in);
    DataflowBuffer cb_pad(dfb::pad);

    cb_in0.reserve_back(num_input_rows);
    cb_in1.reserve_back(num_padded_tiles_per_batch);
    cb_pad.reserve_back(1);

    uint32_t read_addr = cb_in0.get_read_ptr();
    uint32_t write_addr = cb_in1.get_write_ptr();
    uint32_t pad_addr = cb_pad.get_write_ptr();

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
    cb_in1.push_back(num_padded_tiles_per_batch);

    for (uint32_t b = 1; b < num_batches; ++b) {
        cb_in1.reserve_back(num_padded_tiles_per_batch);
        write_addr = cb_in1.get_write_ptr();
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
        cb_in1.push_back(num_padded_tiles_per_batch);
    }
}
