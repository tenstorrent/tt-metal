// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/pause.h"

#include <cstdint>

using sem_ptr_t = volatile tt_l1_ptr uint32_t*;

constexpr uint32_t ilog2(uint32_t n) { return 31 - __builtin_clz(n); }

/**

* Exchange Wt tiles between two cores
*
* Read Wt tiles from value_tensor_cb_index, sent them to peer
* Then, receive Wt tiles from peer and write them into cb_other_index
 */
FORCE_INLINE
void sort_noc_exchange_Wt_tiles(
    uint32_t value_tensor_this_cb_index,
    uint32_t index_tensor_this_cb_index,
    uint32_t cb_value_peer_index,
    uint32_t cb_index_peer_index,
    uint32_t Wt,
    uint32_t value_cb_tile_size,
    uint32_t index_cb_tile_size,
    uint32_t other_core_x,
    uint32_t other_core_y,
    sem_ptr_t sem_self_ptr) {
    constexpr uint32_t ONE_TILE = 1;

    const uint64_t sem_noc_addr = get_noc_addr(other_core_x, other_core_y, reinterpret_cast<uint32_t>(sem_self_ptr));

    for (uint32_t w = 0, sem_counter = 1; w < Wt; w++, sem_counter += 2) {
        // Received tile, now sending it to compute

        DPRINT << "READER: exchanging tile " << w << "/" << Wt << ENDL();
        cb_reserve_back(cb_value_peer_index, ONE_TILE);
        cb_reserve_back(cb_index_peer_index, ONE_TILE);

        uint32_t cb_value_peer_local_write_addr = get_write_ptr(cb_value_peer_index);
        uint32_t cb_index_peer_local_write_addr = get_write_ptr(cb_index_peer_index);

        uint64_t cb_value_peer_noc_write_addr =
            get_noc_addr(other_core_x, other_core_y, cb_value_peer_local_write_addr);
        uint64_t cb_index_peer_noc_write_addr =
            get_noc_addr(other_core_x, other_core_y, cb_index_peer_local_write_addr);

        // Handshake for tile exchange
        noc_semaphore_inc(sem_noc_addr, 1);
        noc_semaphore_wait(sem_self_ptr, sem_counter);

        // Send local indices and values to peer
        DPRINT << "READER: waiting for own input " << w << "/" << Wt << ENDL();
        cb_wait_front(value_tensor_this_cb_index, ONE_TILE);
        DPRINT << "READER: waiting for own indices " << w << "/" << Wt << ENDL();
        cb_wait_front(index_tensor_this_cb_index, ONE_TILE);
        uint32_t value_cb_self_read_addr = get_read_ptr(value_tensor_this_cb_index);
        uint32_t index_cb_self_read_addr = get_read_ptr(index_tensor_this_cb_index);

        noc_async_write(value_cb_self_read_addr, cb_value_peer_noc_write_addr, value_cb_tile_size);
        noc_async_write(index_cb_self_read_addr, cb_index_peer_noc_write_addr, index_cb_tile_size);

        noc_async_write_barrier();

        cb_pop_front(value_tensor_this_cb_index, ONE_TILE);
        cb_pop_front(index_tensor_this_cb_index, ONE_TILE);

        DPRINT << "READER: increasing semaphore timestamp" << ENDL();
        noc_semaphore_inc(sem_noc_addr, 1);  // increment semaphore timestamp
        noc_semaphore_wait(sem_self_ptr, sem_counter + 1);

        cb_push_back(cb_value_peer_index, ONE_TILE);
        cb_push_back(cb_index_peer_index, ONE_TILE);
    }  // Wt

    noc_semaphore_set(sem_self_ptr, 0);  // reset semaphore timestamp
}

FORCE_INLINE
void sort_barrier() {}

FORCE_INLINE std::pair<uint32_t, uint32_t> get_core_physical_coordinates(
    const uint32_t core_id, const uint32_t lookup_table_buffer_cb_index, const uint32_t tile_size = 1024) {
    // Initialize as max to indicate invalid coordinates
    uint32_t core_x = 0;
    uint32_t core_y = 0;

    if (2 * core_id >= tile_size) {
        return {core_x, core_y};  // Invalid core ID
    }

    const uint32_t l1_read_addr = get_read_ptr(lookup_table_buffer_cb_index);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);

    core_x = ptr[core_id * 2];
    core_y = ptr[core_id * 2 + 1];

    return {core_x, core_y};
}
