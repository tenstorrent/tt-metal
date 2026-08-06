// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#ifdef ENABLE_GLOBAL_CB
#include "api/remote_circular_buffer.h"
#endif

// Gathers this core's batch-block rows of width(K)-sharded A into full_in0 (sender-major layout).
//
// in1 (weights) arrive one of two ways:
//   - default: the in1 CB is globally allocated over the L1-resident weight shard, so the
//     reader only has to declare its tiles available.
//   - ENABLE_GLOBAL_CB: the weights are pushed into a DRAM-sender GlobalCircularBuffer by the
//     tensor prefetcher. Exactly one remote page per invocation carries this receiver's whole
//     [Bc*K, Nc] slab. The reader waits for that page, hands the tiles to compute through the
//     local alias CB, and releases the page only after compute signals (via the sync CB) that
//     it has finished reading -- releasing earlier would let the prefetcher overwrite weights
//     still in use. This kernel runs only on the B core range set, so every core is a receiver.
void kernel_main() {
    constexpr uint32_t in0_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t full_in0_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t block_slice_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);
    constexpr uint32_t in1_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t in1_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t remote_cb_index = get_compile_time_arg_val(7);
    constexpr uint32_t sync_cb_index = get_compile_time_arg_val(8);

    const uint32_t b_idx = get_arg_val<uint32_t>(0);

    const uint32_t block_slice_bytes = block_slice_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer in0_cb(in0_cb_index);
    CircularBuffer in1_cb(in1_cb_index);
    CircularBuffer full_in0_cb(full_in0_cb_index);
    UnicastEndpoint sender;

#ifdef ENABLE_GLOBAL_CB
    in1_cb.reserve_back(in1_num_tiles);
    experimental::remote_cb_wait_front(remote_cb_index, 1);
    in1_cb.push_back(in1_num_tiles);
#else
    in1_cb.reserve_back(in1_num_tiles);
    in1_cb.push_back(in1_num_tiles);
#endif
    full_in0_cb.reserve_back(num_senders * block_slice_tiles);

    // in0 shard is at the same L1 offset on every core.
    const uint32_t src_addr = in0_cb.get_read_ptr() + b_idx * block_slice_bytes;
    for (uint32_t s = 0; s < num_senders; ++s) {
        const uint32_t sender_x = get_arg_val<uint32_t>(1 + 2 * s);
        const uint32_t sender_y = get_arg_val<uint32_t>(2 + 2 * s);
        noc.async_read(
            sender,
            full_in0_cb,
            block_slice_bytes,
            {.noc_x = sender_x, .noc_y = sender_y, .addr = src_addr},
            {.offset_bytes = s * block_slice_bytes});
    }
    noc.async_read_barrier();

    full_in0_cb.push_back(num_senders * block_slice_tiles);

#ifdef ENABLE_GLOBAL_CB
    // Compute signals here once it has finished reading every in1 tile of this slab.
    CircularBuffer sync_cb(sync_cb_index);
    sync_cb.wait_front(1);
    sync_cb.pop_front(1);
    experimental::remote_cb_pop_front(remote_cb_index, 1);
    // Persist the remote read pointer so the next invocation resumes at the right ring offset.
    experimental::update_remote_cb_config_in_l1(remote_cb_index);
    noc.async_atomic_barrier();
#endif
}
