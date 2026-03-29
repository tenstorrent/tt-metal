// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Fill a CB buffer with zeros via loopback NOC writes from the hardware MEM_ZEROS region.
// Writes are preferred over reads for this local copy (no handshake overhead).
// The buffer is filled up to NOC_MAX_BURST_SIZE bytes.
FORCE_INLINE void fill_zero_buffer(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    uint32_t buf = get_write_ptr(cb_id);
    uint64_t buf_noc = get_noc_addr(NOC_X(my_x[0]), NOC_Y(my_y[0]), buf);
    for (uint32_t off = 0; off < NOC_MAX_BURST_SIZE; off += MEM_ZEROS_SIZE) {
        uint32_t chunk = ((uint32_t)MEM_ZEROS_SIZE < (NOC_MAX_BURST_SIZE - off)) ? (uint32_t)MEM_ZEROS_SIZE
                                                                                 : (NOC_MAX_BURST_SIZE - off);
        noc_async_write(MEM_ZEROS_BASE, buf_noc + off, chunk);
    }
    noc_async_write_barrier();
}

// Write zeros to a range of interleaved pages using the pre-filled zero buffer.
template <typename AddrGen>
FORCE_INLINE void zero_pages(
    uint32_t zero_buf, uint32_t page_start, uint32_t page_end, uint32_t page_size, const AddrGen& addr_gen) {
    for (uint32_t page = page_start; page < page_end; page++) {
        uint64_t page_noc_addr = get_noc_addr(page, addr_gen);
        uint32_t remaining = page_size;
        while (remaining > 0) {
            uint32_t chunk = (remaining > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE : remaining;
            noc_async_write(zero_buf, page_noc_addr, chunk);
            page_noc_addr += chunk;
            remaining -= chunk;
        }
    }
    noc_async_write_barrier();
}
