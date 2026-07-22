// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Per-batch route plan shared between the dispatch worker reader and writer kernels.
// The reader builds one plan page per batch; the writer drains it. Both sides view the
// same L1 page through these structs (no magic offsets / hand-packed bit fields).
//
// Page layout (one CB page per batch):
//   [ PlanHeader (16 B) ][ PlanEntry[0] (48 B) ][ PlanEntry[1] (48 B) ] ... [ PlanEntry[entry_count-1] ]
//
// Both structs are alignas(16): the page base (CB write ptr) is 16B-aligned, the header is a
// multiple of 16B, and each entry is a multiple of 16B — so every entry starts on a 16B boundary
// and no field can ever straddle a 16B L1 line, no matter how the layout evolves.
//
// End-of-plan sentinel page: entry_count == 0 and entries[0].flags has PLAN_FLAG_END set.

// L1 access granularity both structs are aligned to, so no field ever straddles an L1 line and
// every entry starts on an L1 boundary. Defined locally (not pulled from noc_params.h) because
// that macro differs between host and kernel includes of this shared header.
constexpr uint32_t PLAN_L1_ALIGNMENT = 16;

// flags bits
constexpr uint32_t PLAN_FLAG_LOCAL = 0x1u;       // entry targets the local device
constexpr uint32_t PLAN_FLAG_END = 0x80000000u;  // end-of-plan sentinel (first entry of final page)

// Page header (alignas rounds sizeof up to PLAN_L1_ALIGNMENT); entries follow immediately after.
struct alignas(PLAN_L1_ALIGNMENT) PlanHeader {
    uint32_t entry_count;
};

// Routing entry: 9 × u32 of payload, alignas rounds sizeof up to 48 B (12 B trailing pad)
// so every entry starts on an L1 line boundary.
//
// NOTE: every field is a 32-bit word and the struct is built directly in L1 by the reader.
// Baby-RISC (BRISC/NCRISC) sub-word stores to L1 — `sh` (half-word) / `sb` (byte) — are
// unreliable on Blackhole (unaligned half-words corrupt / drop a byte across the 16B boundary;
// only aligned 32-bit `sw` stores are safe). So weight (signed) + k (top-k slot) are packed
// into a single 32-bit word `weight_k` and written/read via the helpers below — never as
// separate 16-bit fields.
struct alignas(PLAN_L1_ALIGNMENT) PlanEntry {
    uint32_t flags;    // PLAN_FLAG_* bits (bit 0 = is_local, bit 31 = end sentinel)
    uint32_t token_t;  // offset within the batch (0 .. read_batch_size - 1)
    uint32_t routed_expert;
    uint32_t page_idx;   // destination DRAM page (local + remote)
    uint32_t token_idx;  // global token index, for metadata
    uint32_t weight_k;   // packed: low 16 bits = int16_t routing weight, high 16 bits = uint16_t top-k slot
    uint32_t route;      // cross-device only (1D EDM index)
    uint32_t distance;   // cross-device only (1D hop count)
    uint32_t dst_chip;   // cross-device only (linearized dest device index; consumed by the 2D fabric route)
};

static_assert(sizeof(PlanHeader) == PLAN_L1_ALIGNMENT, "PlanHeader must be 1 u32 padded up to PLAN_L1_ALIGNMENT");
static_assert(alignof(PlanHeader) == PLAN_L1_ALIGNMENT, "PlanHeader must be PLAN_L1_ALIGNMENT-aligned");
static_assert(sizeof(PlanEntry) == 48, "PlanEntry must be 48 bytes (9 u32 + 12B pad for 16B alignment)");
static_assert(alignof(PlanEntry) == PLAN_L1_ALIGNMENT, "PlanEntry must be PLAN_L1_ALIGNMENT-aligned");

// weight (signed) + k packed into one 32-bit word so the reader emits a single aligned L1 store.
inline uint32_t pack_weight_k(int16_t weight, uint16_t k) { return ((uint32_t)(uint16_t)weight) | ((uint32_t)k << 16); }
inline int16_t unpack_weight(uint32_t weight_k) { return (int16_t)(weight_k & 0xFFFFu); }
inline uint16_t unpack_k(uint32_t weight_k) { return (uint16_t)(weight_k >> 16); }
