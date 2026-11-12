// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t input_width_bytes = get_arg(args::input_width_bytes);
    const uint32_t input_block_size = get_arg(args::input_block_size);
    const uint32_t num_padded_tiles_per_batch = get_arg(args::num_padded_tiles_per_batch);
    const uint32_t num_padded_rows = get_arg(args::num_padded_rows);
    const uint32_t num_batches = get_arg(args::num_batches);
    const uint32_t packed_pad_value = get_arg(args::packed_pad_value);

    Noc noc;
    // src_shard is a LocalTensorAccessor over this core's borrowed input shard (L1); read-only here.
    // dfb_in1 is the row-major staging DFB the compute kernel tilizes from.
    // pad holds one row of the pad value, reused for every padded row. It is reader-private with no
    // second party, so the former self-loop DFB (bound PRODUCER+CONSUMER) synchronized nothing;
    // converted to a Scratchpad.
    LocalTensorAccessor<uint32_t> src_shard(tensor::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    Scratchpad<volatile uint32_t> pad(scratch::pad);

    dfb_in1.reserve_back(num_padded_tiles_per_batch);

    // Byte offset into src_shard; src_shard and pad are read as loop-back NoC sources (this core's own L1).
    uint32_t read_offset = 0;
    uint32_t write_addr = dfb_in1.get_write_ptr();

    {
        CoreLocalMem<uint32_t> dst(write_addr);
        noc.async_read(src_shard, dst, input_block_size, {.offset_bytes = read_offset}, {.offset_bytes = 0});
    }
    read_offset += input_block_size;
    write_addr += input_block_size;
    {
        // The pad row is CPU-filled here and then used as a NOC read source below. On Quasar the DM
        // core's writes sit in L2 cache; scoped_lock releases (flushes) them so the NOC read sees the
        // filled data instead of stale/zero L1. No-op on Wormhole/Blackhole.
        auto pad_lock = pad.scoped_lock(0, input_width_bytes >> 2);
        for (uint32_t i = 0; i < input_width_bytes >> 2; ++i) {
            pad[i] = packed_pad_value;
        }
    }
    for (uint32_t i = 0; i < num_padded_rows; ++i) {
        CoreLocalMem<uint32_t> dst(write_addr);
        noc.async_read(pad, dst, input_width_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
        write_addr += input_width_bytes;
    }
    noc.async_read_barrier();
    dfb_in1.push_back(num_padded_tiles_per_batch);

    for (uint32_t b = 1; b < num_batches; ++b) {
        dfb_in1.reserve_back(num_padded_tiles_per_batch);
        write_addr = dfb_in1.get_write_ptr();
        {
            CoreLocalMem<uint32_t> dst(write_addr);
            noc.async_read(src_shard, dst, input_block_size, {.offset_bytes = read_offset}, {.offset_bytes = 0});
        }
        read_offset += input_block_size;
        write_addr += input_block_size;
        for (uint32_t i = 0; i < num_padded_rows; ++i) {
            CoreLocalMem<uint32_t> dst(write_addr);
            noc.async_read(pad, dst, input_width_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
            write_addr += input_width_bytes;
        }
        noc.async_read_barrier();
        dfb_in1.push_back(num_padded_tiles_per_batch);
    }
}
