// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One operand stream: a sender reads each block into its own ring slot and multicasts it into the same slot of
// every receiver. Both counters only grow, and every wait is a wait_min:
//   granted (on the sender): +1 from each receiver once its slot for the next block is free;
//   arrived (on the receivers): set by the sender to the number of blocks it has delivered.
// Every core pushes the same blocks of the same size in the same order, so block b sits at the same L1 address on
// every core. A receiver grants block b + 1 only once block b has arrived, so no block or flag overtakes another.

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"

namespace pirotski_matmul {

// The receivers' rectangle in NoC coordinates, start and end ordered for the stream's NOC.
struct Multicast {
    uint32_t start_x;
    uint32_t start_y;
    uint32_t end_x;
    uint32_t end_y;
    uint32_t count;
};

// After the receiver's reserve_back of the block's slot.
FORCE_INLINE void receive_block(
    const Noc& noc,
    Semaphore<>& granted,
    const Semaphore<>& arrived,
    uint32_t sender_x,
    uint32_t sender_y,
    uint32_t block) {
    granted.up(noc, sender_x, sender_y, 1);
    arrived.wait_min(block + 1);
}

// Multicasts bytes at slot_address into the same address of every receiver, then sets their count to delivered.
// The flush keeps the flag behind the data and leaves the slot free to be refilled once this returns; it also
// retires the previous flag's read of the word set here.
FORCE_INLINE void multicast_block(
    const Noc& noc,
    Semaphore<>& arrived,
    const Multicast& receivers,
    uint32_t slot_address,
    uint32_t bytes,
    uint32_t delivered) {
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(slot_address),
        MulticastEndpoint{},
        bytes,
        receivers.count,
        {},
        {.noc_x_start = receivers.start_x,
         .noc_y_start = receivers.start_y,
         .noc_x_end = receivers.end_x,
         .noc_y_end = receivers.end_y,
         .addr = slot_address},
        true);
    noc.async_writes_flushed();
    arrived.set(delivered);
    arrived.set_multicast(noc, receivers.start_x, receivers.start_y, receivers.end_x, receivers.end_y, receivers.count);
}

// Once the block is in the sender's slot.
FORCE_INLINE void send_block(
    const Noc& noc,
    const Semaphore<>& granted,
    Semaphore<>& arrived,
    const Multicast& receivers,
    uint32_t slot_address,
    uint32_t bytes,
    uint32_t block) {
    granted.wait_min((block + 1) * receivers.count);
    multicast_block(noc, arrived, receivers, slot_address, bytes, block + 1);
}

// The rows of the last output block a core owns: those below its end, up to the block's height.
FORCE_INLINE uint32_t valid_rows(uint32_t first_row, uint32_t end_row, uint32_t block_rows) {
    if (end_row <= first_row) {
        return 0;
    }
    return end_row - first_row < block_rows ? end_row - first_row : block_rows;
}

}  // namespace pirotski_matmul
