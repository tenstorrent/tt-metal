// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rect.hpp"
#include "ring.hpp"

// Queue of messages from one B core into one relay core: relay_slots slots, each a message header followed by the
// payload.
struct RelayPort {
    uint32_t x;
    uint32_t y;
    uint32_t slots;
    uint32_t header_scratch;
    Semaphore<> free;
    uint32_t pushes;

    void push(
        const Noc& noc,
        MessageKind kind,
        uint32_t hops,
        uint32_t payload_address,
        uint32_t payload_bytes,
        uint32_t dst_x,
        uint32_t dst_y,
        uint32_t dst_address,
        uint32_t semaphore_address) {
        if (pushes >= CT_ARG("relay_slots")) {
            free.wait_min(pushes + 1 - CT_ARG("relay_slots"));
        }
        volatile tt_l1_ptr Message* message = reinterpret_cast<volatile tt_l1_ptr Message*>(header_scratch);
        message->kind = kind;
        message->hops = hops;
        message->bytes = payload_bytes;
        message->dst_x = dst_x;
        message->dst_y = dst_y;
        message->dst_address = dst_address;
        message->semaphore_address = semaphore_address;
        message->sequence = pushes + 1;
        const uint32_t slot = slots + (pushes % CT_ARG("relay_slots")) * CT_ARG("relay_slot_bytes");
        if (kind == MessageKind::Data) {
            noc.async_write(
                CoreLocalMem<uint32_t>(payload_address),
                UnicastEndpoint{},
                payload_bytes,
                {},
                {.noc_x = x, .noc_y = y, .addr = slot + MESSAGE_BYTES});
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(header_scratch),
            UnicastEndpoint{},
            MESSAGE_BYTES,
            {},
            {.noc_x = x, .noc_y = y, .addr = slot});
        noc.async_writes_flushed();
        pushes++;
    }
};
