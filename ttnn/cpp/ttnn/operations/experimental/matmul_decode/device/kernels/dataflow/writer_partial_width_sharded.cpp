// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"

// Cross-core K-reduction: each core unicasts its partial to slot k_idx on the base core.
void kernel_main() {
    constexpr uint32_t partial_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t reduce_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t reduce_sem_id = get_compile_time_arg_val(5);

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t base_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t base_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t is_base = get_arg_val<uint32_t>(3);

    constexpr uint32_t block_size_bytes = block_num_tiles * tile_size_bytes;

    Noc noc;
    CircularBuffer partial_cb(partial_cb_index);
    CircularBuffer reduce_cb(reduce_cb_index);
    Semaphore<> reduce_sem(reduce_sem_id);
    UnicastEndpoint base_core;

    if (is_base) {
        reduce_cb.reserve_back(K_blocks * block_num_tiles);
    }

    partial_cb.wait_front(block_num_tiles);

    // reduce_cb is at the same L1 offset on every core.
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
    if (is_base) {
        reduce_sem.wait(K_blocks);
        reduce_cb.push_back(K_blocks * block_num_tiles);
    }
    noc.async_write_barrier();
    noc.async_read_barrier();
}
