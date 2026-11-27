// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Block operations shared by the g6_to_memory_config movers.
//
// No kernel_lib helper moves whole pages between a TensorAccessor and local L1 by contiguous runs
// (local_copy_helpers_dataflow.hpp programs one fixed self-aimed source with a single-packet size;
// every transfer here aims at a different remote bank or core and runs exceed NOC_MAX_BURST_SIZE),
// so the run walk is realized here on raw NoC calls.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_to_memory_config {

// Pages are always moved whole: cutting a bfloat8_b / bfloat4_b page splits a shared-exponent group.
// Pull: remote pages [first_page, first_page + num_pages) -> local L1 at local_addr, consecutive.
// Push: local L1 at local_addr, consecutive -> remote pages [first_page, first_page + num_pages).
// One NoC command per run that is contiguous on the remote end; the multi-packet entry points split
// runs above NOC_MAX_BURST_SIZE. No barrier: the caller drains once per block.
template <bool kPull, typename Remote>
FORCE_INLINE void move_span(
    const Remote& remote, uint32_t first_page, uint32_t num_pages, uint32_t local_addr, uint32_t page_size) {
    // An interleaved end, or a shard one page wide, has no two consecutive page ids adjacent in memory.
    const bool remote_runs = remote.contiguous_page_stride() == 1;
    const uint32_t end_page = first_page + num_pages;
    uint32_t page = first_page;
    while (page < end_page) {
        const uint32_t run = remote_runs ? remote.num_contiguous_pages(page, end_page) : 1;
        const uint32_t bytes = run * page_size;
        const uint64_t remote_addr = remote.get_noc_addr(page);
        if constexpr (kPull) {
            noc_async_read(remote_addr, local_addr, bytes);
        } else {
            noc_async_write(local_addr, remote_addr, bytes);
        }
        page += run;
        local_addr += bytes;
    }
}

FORCE_INLINE bool is_this_core(uint64_t noc_addr) {
    return NOC_UNICAST_ADDR_X(noc_addr) == my_x[noc_index] && NOC_UNICAST_ADDR_Y(noc_addr) == my_y[noc_index];
}

FORCE_INLINE uint32_t local_part(uint64_t noc_addr) {
    return static_cast<uint32_t>(noc_addr & ((uint64_t{1} << NOC_ADDR_LOCAL_BITS) - 1));
}

// L1 bank ids are a shuffle of the worker grid that the host does not see, so each core finds its own
// bank by comparing every bank's NoC coordinates (on this kernel's NoC) with its own.
// Returns NUM_L1_BANKS when this core holds no L1 bank.
template <typename L1Interleaved>
FORCE_INLINE uint32_t find_own_l1_bank(const L1Interleaved& owner) {
    for (uint32_t bank = 0; bank < NUM_L1_BANKS; ++bank) {
        if (is_this_core(owner.get_noc_addr(bank))) {
            return bank;
        }
    }
    return NUM_L1_BANKS;
}

}  // namespace g6_to_memory_config
