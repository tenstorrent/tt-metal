// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto stick_size = get_arg(args::stick_size);
    const auto stick_size_offset = get_arg(args::stick_size_offset);
    const auto num_sticks_per_core = get_arg(args::num_sticks_per_core);
    const auto num_sticks_per_core_read = get_arg(args::num_sticks_per_core_read);
    const auto num_read_per_barrier = get_arg(args::num_read_per_barrier);
    const auto start_id = get_arg(args::start_id);
    // Sub-row chunking (mirrors reader): `num_chunks_per_stick` DFB entries of `chunk_size` per stick (last =
    // `last_chunk_size`).
    const auto chunk_size = get_arg(args::chunk_size);
    const auto num_chunks_per_stick = get_arg(args::num_chunks_per_stick);
    const auto last_chunk_size = get_arg(args::last_chunk_size);

    // The per-page stride is the buffer's aligned page size, which the binding supplies; the host
    // pins that it equals the per-shard page size this kernel needs (check_accessor_page_size).
    const auto s0 = TensorAccessor(tensor::dst);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer dfb_out0(dfb::out0);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;

    if (num_chunks_per_stick > 1) {
        // Chunked path (mirrors reader): batch by `num_read_per_barrier` so the second DFB pair pipelines.
        for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
            uint32_t c = 0;
            while (c < num_chunks_per_stick) {
                uint32_t batch = num_chunks_per_stick - c;
                if (batch > num_read_per_barrier) {
                    batch = num_read_per_barrier;
                }
                dfb_out0.wait_front(batch);
                uint32_t l1_read_addr = dfb_out0.get_read_ptr();
                for (uint32_t k = 0; k < batch; ++k) {
                    const uint32_t cur = c + k;
                    const uint32_t offset = cur * chunk_size;
                    const uint32_t sz = (cur == num_chunks_per_stick - 1) ? last_chunk_size : chunk_size;
                    tt::data_movement::common::noc_async_write_sharded(noc, l1_read_addr, s0, i_stick, offset, sz);
                    l1_read_addr += chunk_size;
                }
                noc.async_write_barrier();
                dfb_out0.pop_front(batch);
                c += batch;
            }
            sticks_read++;
            i_stick += 1;
        }
        return;
    }

    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
        dfb_out0.wait_front(num_read_per_barrier);
        uint32_t l1_read_addr = dfb_out0.get_read_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            // noc_async_write_sharded splits the write across shards for B/W-sharded outputs;
            // falls through to a single noc_async_write for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_write_sharded(
                noc, l1_read_addr, s0, i_stick, /*offset=*/0, /*size=*/stick_size);
            l1_read_addr += stick_size_offset;
            i_stick += 1;
        }
        noc.async_write_barrier();
        dfb_out0.pop_front(num_read_per_barrier);
    }
}
