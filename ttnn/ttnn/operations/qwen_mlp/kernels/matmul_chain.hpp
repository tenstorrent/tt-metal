// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "named_args.hpp"

struct Chain {
    Semaphore<> sender;
    Semaphore<> receiver;
    Semaphore<> valid;
    uint32_t dest_x;
    uint32_t dest_y;
    uint32_t prev_x;
    uint32_t prev_y;
    bool is_sink;
    bool is_injector;

    void receive(const Noc& noc) {
        receiver.set(INVALID);
        sender.up(noc, prev_x, prev_y, 1);
        receiver.wait(VALID);
    }

    void forward(const Noc& noc, uint32_t addr, uint32_t bytes) {
        sender.wait(1);
        sender.set(0);
        noc.async_write(
            CoreLocalMem<uint32_t>(addr),
            UnicastEndpoint{},
            bytes,
            {},
            {.noc_x = dest_x, .noc_y = dest_y, .addr = addr});
        noc.async_writes_flushed();
        valid.relay_unicast(noc, receiver, dest_x, dest_y);
    }
};

#define CHAIN(name)                                   \
    Chain{                                            \
        Semaphore<>(CT_ARG("sem_" name "_sender")),   \
        Semaphore<>(CT_ARG("sem_" name "_receiver")), \
        Semaphore<>(CT_ARG("sem_" name "_valid")),    \
        RT_ARG(name "_dest_x"),                       \
        RT_ARG(name "_dest_y"),                       \
        RT_ARG(name "_prev_x"),                       \
        RT_ARG(name "_prev_y"),                       \
        RT_ARG(name "_is_sink") == 1,                 \
        RT_ARG(name "_is_injector") == 1}
