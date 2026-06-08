// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Per-batch route plan shared between the dispatch untilize reader and writer kernels.
// The reader builds one plan page per batch; the writer drains it. Both sides view the
// same L1 page through these structs (no magic offsets / hand-packed bit fields).
//
// Page layout (one CB page per batch):
//   [ PlanHeader (32 B) ][ PlanEntry[0] ][ PlanEntry[1] ] ... [ PlanEntry[entry_count-1] ]
//
// End-of-plan sentinel page: entry_count == 0 and entries[0].flags has PLAN_FLAG_END set.

// flags bits
constexpr uint32_t PLAN_FLAG_LOCAL = 0x1u;       // entry targets the local device
constexpr uint32_t PLAN_FLAG_END = 0x80000000u;  // end-of-plan sentinel (first entry of final page)

// 32-byte page header; entries follow immediately after.
struct PlanHeader {
    uint32_t entry_count;
    uint32_t reserved[7];
};

// 36-byte (9 × u32) routing entry.
struct PlanEntry {
    uint32_t flags;    // PLAN_FLAG_* bits (bit 0 = is_local, bit 31 = end sentinel)
    uint32_t token_t;  // offset within the batch (0 .. read_batch_size - 1)
    uint32_t routed_expert;
    uint32_t page_idx;   // destination DRAM page (local + remote)
    uint32_t token_idx;  // global token index, for metadata
    int16_t weight;      // routing weight (signed); packed low 16 bits of the legacy [5] word
    uint16_t k;          // top-k slot;                 packed high 16 bits of the legacy [5] word
    uint32_t route;      // cross-device only (1D EDM index)
    uint32_t distance;   // cross-device only (1D hop count)
    uint32_t dst_chip;   // cross-device only (linearized dest device index; consumed by the 2D fabric route)
};

static_assert(sizeof(PlanHeader) == 8 * sizeof(uint32_t), "PlanHeader must be 32 bytes");
static_assert(sizeof(PlanEntry) == 9 * sizeof(uint32_t), "PlanEntry must be 36 bytes");
