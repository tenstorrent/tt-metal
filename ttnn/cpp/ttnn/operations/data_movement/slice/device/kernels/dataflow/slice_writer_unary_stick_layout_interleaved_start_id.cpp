// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t stick_size = get_arg_val<uint32_t>(1);
    uint32_t stick_size_offset = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_core = get_arg_val<uint32_t>(3);
    uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(4);
    uint32_t num_read_per_barrier = get_arg_val<uint32_t>(5);
    uint32_t start_id = get_arg_val<uint32_t>(6);
    // Per-shard page size: shard_W on B/W-sharded outputs, full row otherwise.
    // Feeds `noc_async_write_sharded`'s multi-shard split via `get_aligned_page_size()`.
    uint32_t page_size_override = get_arg_val<uint32_t>(7);
    // Sub-row chunking (mirrors reader): `num_chunks_per_stick` CB pages of `chunk_size` per stick (last =
    // `last_chunk_size`).
    const uint32_t chunk_size = get_arg_val<uint32_t>(8);
    const uint32_t num_chunks_per_stick = get_arg_val<uint32_t>(9);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(10);

    constexpr uint32_t dfb_id_out0 = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    const auto s0 = TensorAccessor(dst_args, dst_addr, page_size_override);

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer dfb_out0(dfb_id_out0);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;

    if (num_chunks_per_stick > 1) {
        // Chunked path (mirrors reader): batch by `num_read_per_barrier` so the second CB pair pipelines.
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
