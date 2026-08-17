// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

template <bool needs_zeroing = true>
FORCE_INLINE void wh_generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler) {
    // This is much faster but WILL NOT WORK IN BLACKHOLE since it assumes 32B alignment noc reads are allowed, done
    // for llama effort
    cb_reserve_back(cb_id, 1);
    uint32_t write_addr_base = get_write_ptr(cb_id);
    uint64_t target_address = get_noc_addr(write_addr_base);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr_base);

    // Fill tile with zeros
    if constexpr (needs_zeroing) {
        Noc noc;
        CircularBuffer cb(cb_id);
        noc.async_write_zeros(cb, 2048);
        noc.write_zeros_l1_barrier();
    }

    if (scaler != 0 || !needs_zeroing) {
        for (int j = 0; j < 8; ++j) {
            ptr[j] = scaler;
        }
        noc_async_read_one_packet_set_state(target_address, 32);
        noc_async_read_one_packet_with_state(target_address, write_addr_base + (1 << 9));
        noc_async_read_one_packet_with_state(target_address, write_addr_base + (2 << 9));
        noc_async_read_one_packet_with_state(target_address, write_addr_base + (3 << 9));
        noc_async_read_barrier();
    }
    cb_push_back(cb_id, 1);
}
