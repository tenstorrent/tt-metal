// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"

// Copies kv_cache rows rows[first, last) to output rows [output_offset + first, output_offset + last).
// There is no direct DRAM -> remote L1 path, so each row is staged in this core's L1. The stage is split
// into two banks: one bank is read into while the other bank's writes drain.
// stage is kernel-local scratch with stage_rows entries of one row each; it is reserved and never pushed.
template <uint32_t stage_rows, typename KvCache, typename Output>
FORCE_INLINE void gather_kv_rows(
    const Noc& noc,
    const KvCache& kv_cache,
    const Output& output,
    DataflowBuffer& stage,
    volatile tt_l1_ptr uint32_t* rows,
    uint32_t first,
    uint32_t last,
    uint32_t output_offset) {
    static_assert(stage_rows >= 2 && stage_rows % 2 == 0, "stage_rows must be even so it splits into two banks");
    constexpr uint32_t bank_rows = stage_rows / 2;
    const uint32_t row_bytes = stage.get_entry_size();
    stage.reserve_back(stage_rows);
    uint32_t bank = 0;
    for (uint32_t row = first; row < last; row += bank_rows, bank ^= 1) {
        const uint32_t n = std::min(bank_rows, last - row);
        const uint32_t base = bank * bank_rows;
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_read(
                kv_cache, stage, row_bytes, {.page_id = rows[row + j]}, {.offset_bytes = (base + j) * row_bytes});
        }
        noc.async_read_barrier();
        // Drains the other bank's writes, which were overlapping these reads, so it can be read into next.
        noc.async_writes_flushed();
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_write(
                stage,
                output,
                row_bytes,
                {.offset_bytes = (base + j) * row_bytes},
                {.page_id = output_offset + row + j});
        }
    }
    noc.async_write_barrier();
}
