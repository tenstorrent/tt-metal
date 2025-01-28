// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <vector>

#include "debug/dprint.h"  // required in all kernels using DPRINT

template <bool is_dram, uint32_t priority_scalar_bytes, uint32_t B, uint32_t cb_scratch_id>
std::pair<uint32_t, uint32_t> get_max_priority(uint32_t priority_addr, uint32_t other_priority_addr) {
    // reserve scratch space
    const InterleavedAddrGen<is_dram> priority_addr_gen = {
        .bank_base_address = priority_addr, .page_size = priority_scalar_bytes};
    const InterleavedAddrGen<is_dram> other_priority_addr_gen = {
        .bank_base_address = other_priority_addr, .page_size = priority_scalar_bytes};

    DPRINT << "priority_scalar_bytes: " << priority_scalar_bytes << ENDL();

    uint32_t max_priority = 0;
    for (uint32_t i = 0; i < B; i++) {
        cb_reserve_back(cb_scratch_id, 1);

        uint32_t priority_cb_wr_ptr = get_write_ptr(cb_scratch_id);
        uint64_t priority_noc_addr = get_noc_addr(i, priority_addr_gen);
        noc_async_read(priority_noc_addr, priority_cb_wr_ptr, priority_scalar_bytes);
        DPRINT << "priority_cb_wr_ptr: " << priority_cb_wr_ptr << ENDL();
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* priority_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(priority_cb_wr_ptr);
        uint32_t cur_priority = priority_ptr[0];
        DPRINT << "cur_priority: " << cur_priority << ENDL();
        if (cur_priority > max_priority) {
            max_priority = cur_priority;
        }

        cb_push_back(cb_scratch_id, 1);
    }

    uint32_t max_other_priority = 0;
    for (uint32_t i = 0; i < B; i++) {
        cb_reserve_back(cb_scratch_id, 1);

        uint32_t other_priority_cb_wr_ptr = get_write_ptr(cb_scratch_id);
        uint64_t other_priority_noc_addr = get_noc_addr(i, other_priority_addr_gen);
        noc_async_read(other_priority_noc_addr, other_priority_cb_wr_ptr, priority_scalar_bytes);
        DPRINT << "other_priority_cb_wr_ptr: " << other_priority_cb_wr_ptr << ENDL();
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* other_priority_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(other_priority_cb_wr_ptr);
        uint32_t cur_other_priority = other_priority_ptr[0];
        DPRINT << "cur_other_priority: " << cur_other_priority << ENDL();
        if (cur_other_priority > max_other_priority) {
            max_other_priority = cur_other_priority;
        }

        cb_push_back(cb_scratch_id, 1);
    }

    return std::make_pair(max_priority, max_other_priority);
}

template <uint32_t cb_scratch_id, uint32_t B, uint32_t priority_stick_size>
std::pair<uint32_t, uint32_t> read_max_priority_from_scratch() {
    uint32_t cum_wait = 0;
    uint32_t max_priority = 0;
    for (uint32_t i = 0; i < B; i++) {
        cum_wait += 1;
        cb_wait_front(cb_scratch_id, cum_wait);

        uint32_t read_offset = priority_stick_size * i;
        uint32_t priority_cb_ptr = get_read_ptr(cb_scratch_id) + read_offset;
        volatile tt_l1_ptr uint32_t* priority_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(priority_cb_ptr);
        uint32_t cur_priority = priority_ptr[0];
        DPRINT << "cur_priority: " << cur_priority << ENDL();
        if (cur_priority > max_priority) {
            max_priority = cur_priority;
        }
    }

    uint32_t max_other_priority = 0;
    for (uint32_t i = 0; i < B; i++) {
        cum_wait += 1;
        cb_wait_front(cb_scratch_id, cum_wait);

        uint32_t read_offset = priority_stick_size * (i + B);
        uint32_t other_priority_cb_ptr = get_read_ptr(cb_scratch_id) + read_offset;
        volatile tt_l1_ptr uint32_t* other_priority_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(other_priority_cb_ptr);
        uint32_t cur_other_priority = other_priority_ptr[0];
        DPRINT << "cur_other_priority: " << cur_other_priority << ENDL();
        if (cur_other_priority > max_other_priority) {
            max_other_priority = cur_other_priority;
        }
    }

    return std::make_pair(max_priority, max_other_priority);
}
