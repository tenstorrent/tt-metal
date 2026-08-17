// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

// Zero a range of interleaved DRAM pages via the device-side zero-memory API.
//
// A NOC_MAX_BURST_SIZE L1 scratch (the supplied CB) is pre-zeroed once, then written to
// each DRAM page, which chunks any page larger than NOC_MAX_BURST_SIZE internally. The
// scratch CB must be at least NOC_MAX_BURST_SIZE bytes.
template <typename AddrGen>
FORCE_INLINE void zero_pages(
    uint32_t scratch_cb_id, uint32_t page_start, uint32_t page_end, uint32_t page_size, const AddrGen& addr_gen) {
    Noc noc;
    CircularBuffer scratch_cb(scratch_cb_id);
    noc.async_write_zeros(scratch_cb, NOC_MAX_BURST_SIZE);
    noc.write_zeros_l1_barrier();
    for (uint32_t page = page_start; page < page_end; page++) {
        noc.async_write_zeros(addr_gen, page_size, {.page_id = page}, scratch_cb);
    }
    noc.write_zeros_dram_barrier();
}
