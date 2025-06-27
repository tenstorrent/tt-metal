// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/pause.h"

#include <cstdint>

using sem_ptr_t = volatile tt_l1_ptr uint32_t*;

void sort_noc_exchange_Wt_tiles(
    uint32_t cb_other_index,
    uint32_t value_tensor_cb_index,
    uint32_t Wt,
    uint32_t value_cb_tile_size,
    uint32_t other_core_x,
    uint32_t other_core_y,
    sem_ptr_t sem_self_ptr,
    uint64_t sem_noc_addr) {
    constexpr uint32_t ONE_TILE = 1;

    for (uint32_t w = 0, sem_counter = 0; w < Wt; w++, sem_counter += 2) {
        // Received tile, now sending it to compute
        cb_reserve_back(cb_other_index, ONE_TILE);
        uint32_t other_cb_self_write_addr = get_write_ptr(cb_other_index);
        uint64_t other_cb_noc_write_addr = get_noc_addr(other_core_x, other_core_y, other_cb_self_write_addr);

        // Handshake for tile exchange
        noc_semaphore_inc(sem_noc_addr, 1);

        noc_semaphore_wait(sem_self_ptr, sem_counter);

        cb_wait_front(value_tensor_cb_index, ONE_TILE);
        uint32_t value_cb_self_read_addr = get_read_ptr(value_tensor_cb_index);

        noc_async_write(value_cb_self_read_addr, other_cb_noc_write_addr, value_cb_tile_size);
        noc_async_write_barrier();

        cb_pop_front(value_tensor_cb_index, ONE_TILE);

        noc_semaphore_inc(sem_noc_addr, 1);
        noc_semaphore_wait(sem_self_ptr, sem_counter + 1);

        cb_push_back(cb_other_index, ONE_TILE);
    }

    noc_semaphore_set(sem_self_ptr, 0);
}
