// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

// Inputs that do not depend on the value columns are identical for every value block of a head. With
// mcast_shared, value block 0 reads them from DRAM once and multicasts them into its siblings' buffers.
// The slot addresses match on every core because every block makes the same reserve/push sequence on
// identically specified buffers. Handshake: receivers reserve their slots, reset valid and raise ready; the
// sender stages its reads, waits for every receiver, multicasts the data, flushes, then multicasts the valid
// flag before publishing its own copy.
struct SharedInput {
    DataflowBuffer* buffer;
    uint32_t tiles;
    uint32_t slot;
};

template <typename Accessor>
FORCE_INLINE uint32_t
stage_contiguous_tiles(const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t base, uint32_t count) {
    buffer.reserve_back(count);
    const uint32_t entry_size = buffer.get_entry_size();
    uint32_t tile = 0;
    for (const auto& page : accessor.pages(base, base + count)) {
        noc.async_read(accessor, buffer, entry_size, {.page_id = page.page_id()}, {.offset_bytes = tile * entry_size});
        ++tile;
    }
    return buffer.get_write_ptr();
}

template <uint32_t N, typename ReadySem, typename ValidSem>
FORCE_INLINE void multicast_shared(
    Noc& noc,
    SharedInput (&inputs)[N],
    ReadySem& ready,
    ValidSem& valid,
    uint32_t x0,
    uint32_t y0,
    uint32_t x1,
    uint32_t y1,
    uint32_t receivers) {
    noc.async_read_barrier();
    ready.wait(receivers);
    ready.set(0);
    MulticastEndpoint destination;
    for (auto& input : inputs) {
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(input.slot),
            destination,
            input.tiles * input.buffer->get_entry_size(),
            receivers,
            {},
            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = input.slot},
            /*linked=*/true);
    }
    // The flag multicast issues from a different command buffer than the data, so only this flush orders the data
    // before the flag. It also proves the source slots were read before this core's compute may pop them.
    noc.async_writes_flushed();
    valid.set_multicast(noc, x0, y0, x1, y1, receivers);
    for (auto& input : inputs) {
        input.buffer->push_back(input.tiles);
    }
}

template <uint32_t N, typename ReadySem, typename ValidSem>
FORCE_INLINE void receive_shared(
    Noc& noc, SharedInput (&inputs)[N], ReadySem& ready, ValidSem& valid, uint32_t sender_x, uint32_t sender_y) {
    for (auto& input : inputs) {
        input.buffer->reserve_back(input.tiles);
    }
    // Reset valid before raising ready: a fast sender may multicast VALID right after the increment.
    valid.set(0);
    ready.up(noc, sender_x, sender_y, 1);
    valid.wait(1);
    for (auto& input : inputs) {
        input.buffer->push_back(input.tiles);
    }
}
