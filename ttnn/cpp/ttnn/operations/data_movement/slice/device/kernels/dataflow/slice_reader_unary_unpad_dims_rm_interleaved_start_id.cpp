// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t padded_stick_size = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_stick_size = get_arg_val<uint32_t>(2);
    const uint32_t stick_size_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_dims = get_arg_val<uint32_t>(4);
    const uint32_t misalignment = get_arg_val<uint32_t>(5);
    const uint32_t start_id = get_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_core = get_arg_val<uint32_t>(7);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(8);
    const uint32_t num_read_per_barrier = get_arg_val<uint32_t>(9);
    // Sub-row chunking: `num_chunks_per_stick` NOC transfers of `chunk_size` per stick (last = `last_chunk_size`).
    const uint32_t chunk_size = get_arg_val<uint32_t>(10);
    const uint32_t num_chunks_per_stick = get_arg_val<uint32_t>(11);
    const uint32_t last_chunk_size = get_arg_val<uint32_t>(12);

    tt_l1_ptr uint32_t* num_unpadded_sticks = (tt_l1_ptr uint32_t*)(get_arg_addr(13));
    volatile tt_l1_ptr uint32_t* num_padded_sticks = num_unpadded_sticks + num_dims;
    volatile tt_l1_ptr uint32_t* id_per_dim = num_padded_sticks + num_dims;

    constexpr auto src_args = TensorAccessorArgs<0>();
    uint32_t read_size = unpadded_stick_size + misalignment;

    // padded_stick_size = per-shard page size (shard_W on B/W-sharded, full row otherwise);
    // feeds `noc_async_read_sharded`'s multi-shard split via `get_aligned_page_size()`.
    const auto s0 = TensorAccessor(src_args, src_addr, padded_stick_size);

    constexpr uint32_t dfb_id_in0 = 0;

    Noc noc;
    // Create DataflowBuffer for Device 2.0 API
    DataflowBuffer dfb_in0(dfb_id_in0);

    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;

    if (num_chunks_per_stick > 1) {
        // Chunked path: batch by `num_read_per_barrier` so the second CB pair pipelines behind the first.
        for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
            uint32_t c = 0;
            while (c < num_chunks_per_stick) {
                uint32_t batch = num_chunks_per_stick - c;
                if (batch > num_read_per_barrier) {
                    batch = num_read_per_barrier;
                }
                dfb_in0.reserve_back(batch);
                uint32_t src_buffer_l1_addr = dfb_in0.get_write_ptr();
                for (uint32_t k = 0; k < batch; ++k) {
                    const uint32_t cur = c + k;
                    const uint32_t offset = cur * chunk_size;
                    const uint32_t sz = (cur == num_chunks_per_stick - 1) ? last_chunk_size : chunk_size;
                    tt::data_movement::common::noc_async_read_sharded(
                        noc, src_buffer_l1_addr, s0, src_stick_id, offset, sz);
                    src_buffer_l1_addr += chunk_size;
                }
                noc.async_read_barrier();
                dfb_in0.push_back(batch);
                c += batch;
            }
            sticks_read++;
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        return;
    }

    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
        dfb_in0.reserve_back(num_read_per_barrier);
        uint32_t src_buffer_l1_addr = dfb_in0.get_write_ptr();

        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            // noc_async_read_sharded splits the read across shards for B/W-sharded inputs;
            // falls through to a single noc_async_read for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_read_sharded(
                noc, src_buffer_l1_addr, s0, src_stick_id, /*offset=*/0, /*size=*/read_size);
            if (misalignment != 0) {
                noc.async_read_barrier();
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    noc, src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size);
            }
            src_buffer_l1_addr += stick_size_offset;
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks[j];
                } else {
                    break;
                }
            }
        }
        noc.async_read_barrier();
        dfb_in0.push_back(num_read_per_barrier);
    }
}
