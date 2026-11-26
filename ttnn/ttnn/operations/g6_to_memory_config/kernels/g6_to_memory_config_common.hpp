// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared run-coalescing rule for g6_to_memory_config movers (op_design.md, Mechanism caps):
// a run is a set of consecutive page ids that are also consecutive in the accessor-side memory.
// Only an accessor whose contiguous_page_stride() == 1 can coalesce; an interleaved accessor's
// stride is the bank count (consecutive ids sit in consecutive banks), so its run is always 1.
// The local (resident-shard / CB-slot) side is contiguous by construction over the span the
// caller passes, so the caller bounds the run by passing `end_page` = the span end.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace g6_tmc {

template <typename Accessor>
FORCE_INLINE uint32_t run_pages(const Accessor& acc, uint32_t page, uint32_t end_page) {
    if (acc.contiguous_page_stride() != 1) {
        return 1;
    }
    return acc.num_contiguous_pages(page, end_page);
}

// Read pages [page, page + count) from `src` into local L1 starting at `l1_addr`,
// one NoC read per maximal run. No barrier (the caller settles once per block).
template <typename Accessor>
FORCE_INLINE void read_span(const Accessor& src, uint32_t page, uint32_t count, uint32_t l1_addr, uint32_t page_size) {
    const uint32_t end_page = page + count;
    while (page < end_page) {
        const uint32_t run = run_pages(src, page, end_page);
        const uint32_t bytes = run * page_size;
        noc_async_read(src.get_noc_addr(page), l1_addr, bytes);
        page += run;
        l1_addr += bytes;
    }
}

// Write local L1 starting at `l1_addr` to pages [page, page + count) of `dst`,
// one NoC write per maximal run. No barrier.
template <typename Accessor>
FORCE_INLINE void write_span(const Accessor& dst, uint32_t page, uint32_t count, uint32_t l1_addr, uint32_t page_size) {
    const uint32_t end_page = page + count;
    while (page < end_page) {
        const uint32_t run = run_pages(dst, page, end_page);
        const uint32_t bytes = run * page_size;
        noc_async_write(l1_addr, dst.get_noc_addr(page), bytes);
        page += run;
        l1_addr += bytes;
    }
}

}  // namespace g6_tmc
