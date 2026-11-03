// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

// Cross-core K-reduction: each core unicasts its partial to slot k_idx on the base core.
// IS_BASE: this core is the base of its N block. It collects the K_blocks partials for compute and
// then waits for compute's output, giving that buffer the consumer every buffer needs.
void kernel_main() {
    constexpr uint32_t block_num_tiles = get_arg(args::block_num_tiles);
    constexpr uint32_t tile_size_bytes = get_arg(args::tile_size_bytes);
    constexpr uint32_t K_blocks = get_arg(args::K_blocks);

    const uint32_t k_idx = get_arg(args::k_idx);
    const uint32_t base_noc_x = get_arg(args::base_noc_x);
    const uint32_t base_noc_y = get_arg(args::base_noc_y);

    constexpr uint32_t block_size_bytes = block_num_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer partial_cb(dfb::partial);
    CircularBuffer reduce_cb(dfb::reduce);
    Semaphore<> reduce_sem(sem::reduce);
    UnicastEndpoint base_core;

#ifdef IS_BASE
    reduce_cb.reserve_back(K_blocks * block_num_tiles);
#endif

    partial_cb.wait_front(block_num_tiles);

    // reduce is at the same L1 offset on every weight core.
    const uint32_t dst_addr = reduce_cb.get_write_ptr() + k_idx * block_size_bytes;
    noc.async_write(
        partial_cb,
        base_core,
        block_size_bytes,
        {.offset_bytes = 0},
        {.noc_x = base_noc_x, .noc_y = base_noc_y, .addr = dst_addr});
    noc.async_write_barrier();

    reduce_sem.up(noc, base_noc_x, base_noc_y, 1);
    noc.async_atomic_barrier();

    partial_cb.pop_front(block_num_tiles);
#ifdef IS_BASE
    reduce_sem.wait(K_blocks);
    reduce_cb.push_back(K_blocks * block_num_tiles);

    CircularBuffer out_cb(dfb::out);
    out_cb.wait_front(block_num_tiles);
    out_cb.pop_front(block_num_tiles);
#endif
    noc.async_write_barrier();
    noc.async_read_barrier();
}
