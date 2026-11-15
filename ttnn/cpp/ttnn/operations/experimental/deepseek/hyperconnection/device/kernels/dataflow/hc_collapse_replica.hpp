// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "api/dataflow/dataflow_api.h"

// collapsed's replica (FusedSingleUserParams::collapsed_replica_grid), all_gather_for_matmul's protocol: every
// collapse core writes its slice into the hub's (collapse core 0's) replica row and raises the hub's semaphore; the
// hub, inside the replica rectangle, multicasts the whole row to the rectangle's other cores.
FORCE_INLINE void hc_send_replica_slice(
    uint32_t src_l1,
    uint32_t row_bytes,
    uint32_t core_idx,
    uint32_t replica_addr,
    uint32_t hub_x,
    uint32_t hub_y,
    uint32_t sem_addr,
    uint32_t num_senders,
    uint32_t full_row_bytes,
    uint32_t mcast_start_x,
    uint32_t mcast_start_y,
    uint32_t mcast_end_x,
    uint32_t mcast_end_y,
    uint32_t mcast_cores) {
    noc_async_write(src_l1, get_noc_addr(hub_x, hub_y, replica_addr + core_idx * row_bytes), row_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, sem_addr), 1);
    noc_async_atomic_barrier();
    if (core_idx != 0) {
        return;
    }
    auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, num_senders);
    noc_semaphore_set(sem, 0);
    if (mcast_cores > 1) {
        if (noc_index == 1) {
            std::swap(mcast_start_x, mcast_end_x);
            std::swap(mcast_start_y, mcast_end_y);
        }
        const uint64_t dst =
            get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, replica_addr);
        noc_async_write_multicast(replica_addr, dst, full_row_bytes, mcast_cores - 1);
        noc_async_write_barrier();
    }
}
