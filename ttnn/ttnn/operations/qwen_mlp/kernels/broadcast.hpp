// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "named_args.hpp"

// All-gather inside a column of cores: every block of the stream has one producing core that multicasts it into the
// ring CB of the whole column (itself included). The receivers grant the producer a credit once the slot is free, so
// blocks arrive one after the other and the arrival semaphore can count them.
struct Broadcast {
    Semaphore<> credit;
    Semaphore<> arrival;
    Semaphore<> source;
    uint32_t col_x;
    uint32_t y_first;
    uint32_t y_last;
    uint32_t group_size;
    uint32_t my_row;
    uint32_t row_y_arg;
    uint32_t steps;
    uint32_t sends;

    uint32_t row_y(uint32_t row) const { return get_arg_val<uint32_t>(row_y_arg + row); }

    void multicast(const Noc& noc, uint32_t src, uint32_t dst, uint32_t bytes, Semaphore<>& dst_sem, uint32_t value) {
        noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
            CoreLocalMem<uint32_t>(src),
            MulticastEndpoint{},
            bytes,
            group_size,
            {},
            {.noc_x_start = col_x, .noc_y_start = y_first, .noc_x_end = col_x, .noc_y_end = y_last, .addr = dst},
            true);
        source.set(value);
        source.relay_multicast<NocOptions::MCAST_INCL_SRC>(
            noc, dst_sem, col_x, y_first, col_x, y_last, group_size, false);
        noc.async_writes_flushed();
    }

    template <typename SourceAddress>
    void step(
        const Noc& noc,
        CircularBuffer& ring,
        uint32_t block_tiles,
        uint32_t block_bytes,
        uint32_t producer_row,
        SourceAddress source_address) {
        ring.reserve_back(block_tiles);
        const uint32_t dst = ring.get_write_ptr();
        credit.up(noc, col_x, row_y(producer_row), 1);
        if (producer_row == my_row) {
            sends++;
            credit.wait_min(group_size * sends);
            multicast(noc, source_address(), dst, block_bytes, arrival, steps + 1);
        }
        steps++;
        arrival.wait_min(steps);
        ring.push_back(block_tiles);
    }
};

#define BROADCAST()                                   \
    Broadcast{                                        \
        Semaphore<>(CT_ARG("sem_in0_credit")),        \
        Semaphore<>(CT_ARG("sem_in0_arrival")),       \
        Semaphore<>(CT_ARG("sem_in0_source")),        \
        RT_ARG("col_x"),                              \
        RT_ARG("col_y_0"),                            \
        RT_ARG("col_y_last"),                         \
        CT_ARG("group_size"),                         \
        RT_ARG("group_row"),                          \
        get_named_compile_time_arg_val("rt_col_y_0"), \
        0,                                            \
        0}
