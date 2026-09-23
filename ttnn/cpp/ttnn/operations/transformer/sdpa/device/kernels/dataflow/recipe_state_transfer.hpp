// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/circular_buffer.h"
#include "../recipe_state_layout.hpp"

template <bool fp32, uint32_t request_cb, uint32_t ack_cb, typename Accessor>
void transfer_recipe_state(Noc& noc, const Accessor& backing) {
    using Transfer = sdpa::streaming::StateTransfer;
    CircularBuffer request(request_cb), ack(ack_cb);
    request.wait_front(1);
    ack.reserve_back(1);
    const auto* words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
    const bool restore = words[Transfer::Operation] == Transfer::Restore;
    const uint32_t base = words[Transfer::Slot] * (Transfer::pages<fp32> + 1);
    uint32_t page = base + 1;
    for (uint32_t plane = 0; plane < (fp32 ? 3u : 4u); ++plane) {
        const uint32_t cb = words[Transfer::Numerator + plane];
        const uint32_t bytes = Transfer::plane_bytes<fp32>(plane);
        // These full-capacity state banks are at their allocation origin at
        // every segment boundary. Dataflow never advances their CB pointers.
        const uint32_t address = CircularBuffer(cb).get_read_ptr();
        for (uint32_t offset = 0; offset < bytes; offset += Transfer::page_bytes, ++page) {
            if (restore) {
                noc.async_read(
                    backing, CoreLocalMem<uint32_t>(address + offset), Transfer::page_bytes, {.page_id = page}, {});
            } else {
                noc.async_write(
                    CoreLocalMem<uint32_t>(address + offset), backing, Transfer::page_bytes, {}, {.page_id = page});
            }
        }
    }
    if (restore) {
        noc.async_read(
            backing,
            CoreLocalMem<uint32_t>(ack.get_write_ptr()),
            Transfer::Words * sizeof(uint32_t),
            {.page_id = base},
            {});
        noc.async_read_barrier();
    } else {
        noc.async_write(
            CoreLocalMem<uint32_t>(request.get_read_ptr()),
            backing,
            Transfer::Words * sizeof(uint32_t),
            {},
            {.page_id = base});
        noc.async_write_barrier();
    }
    request.pop_front(1);
    ack.push_back(1);
}
