// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Whole-page moves between one core's L1 and a remote TensorAccessor.
//
// No kernel_lib helper moves pages between a local L1 region and a remote accessor by contiguous
// runs: local_copy_helpers_dataflow programs one fixed self-aimed source and a single-packet size,
// while every run here targets a different remote core/bank and can exceed NOC_MAX_BURST_SIZE.
// These functions are that missing block operation, shared by the owner movers and the staged
// reader/writer.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

namespace g6_to_memory_config {

// Pull: remote -> local L1. Push: local L1 -> remote. Multi-packet entry points only: a run can be a
// whole shard row (up to 128 KB), above the one-packet limit.
template <bool is_pull, typename Remote>
FORCE_INLINE void issue_transfer(const Remote& remote, uint32_t page, uint32_t local_addr, uint32_t bytes) {
    if constexpr (is_pull) {
        noc_async_read(remote.get_noc_addr(page), local_addr, bytes);
    } else {
        noc_async_write(local_addr, remote.get_noc_addr(page), bytes);
    }
}

// Moves remote pages [page, page + num_pages) to/from a contiguous local region, one NoC command per
// run. A run never crosses a remote shard edge; it is one page when the remote's consecutive page ids
// are not adjacent in memory (interleaved, or a one-page-wide shard).
template <bool is_pull, typename Remote>
FORCE_INLINE void move_page_range(
    const Remote& remote, uint32_t page, uint32_t num_pages, uint32_t local_addr, uint32_t page_size) {
    const bool runs_are_contiguous = remote.contiguous_page_stride() == 1;
    while (num_pages > 0) {
        const uint32_t run = runs_are_contiguous ? remote.num_contiguous_pages(page, page + num_pages) : 1;
        const uint32_t bytes = run * page_size;
        issue_transfer<is_pull>(remote, page, local_addr, bytes);
        page += run;
        num_pages -= run;
        local_addr += bytes;
    }
}

}  // namespace g6_to_memory_config
