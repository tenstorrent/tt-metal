// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The combine's data formats: what one routed row carries, the bit-packed words the cores exchange, and the layout of
// the MISC block of every combine core's state. Shared by the host (combine_program.cpp) and the kernels, so each
// format has one definition. Plain constexpr only (no includes beyond <cstdint>): the kernels include it as is.
// The device plan the combine reads (rows per port / accumulator / source) is in
// common/kernels/combine_plan_words.hpp.
#pragma once

#include <cstdint>

namespace combine_formats {

// =====================================================================================================================
// A routed row in a landing slot
// =====================================================================================================================
//
// Every expert output row travels as one packet: the row's bf16 data, then a 32-byte trailer. A landing slot (a ring
// slot in an accumulator's L1, or a pool slot on a flat-expert core) holds exactly that, padded to 64 bytes:
//
//   slot + 0                  slot + row_bytes                 slot + row_bytes + 32        (64 B aligned)
//   +-------------------------+--------------------------------+---------------------------+
//   | row data: W bf16 values | trailer: 8 x uint32            | padding                   |
//   +-------------------------+--------------------------------+---------------------------+
//
// The trailer (written by the flat expert's column-0 y writer, y_row_major.hpp, next to the row's first segment):
//
//   word | contents
//   -----+----------------------------------------------------------------------------------------------
//     0  | header word (below): the token's home chip, the token, the expert
//     1  | the row's gate weight, bf16 bits in bits [15:0]
//     2  | 0
//     3  | arrival flag: 1 once this half of the trailer has landed
//    4-6 | 0
//     7  | arrival flag: 1 once this half of the trailer has landed
//
// The two flags sit in different 16-byte halves of the trailer. A reader takes the row only when both read 1, so it
// never sees a half-written trailer. Taking a row clears both flags (or the whole trailer on a pool slot), so the slot
// reads empty until the next row lands there; the state is zeroed every call, so every slot starts empty.
inline constexpr uint32_t kTrailerBytes = 32;
inline constexpr uint32_t kTrailerHeader = 0;    // trailer word: the header word
inline constexpr uint32_t kTrailerWeight = 1;    // trailer word: the gate weight (bf16 bits)
inline constexpr uint32_t kTrailerFlagLow = 3;   // trailer word: arrival flag in the first 16 bytes
inline constexpr uint32_t kTrailerFlagHigh = 7;  // trailer word: arrival flag in the second 16 bytes

// The header word (the MoE head's routing planner writes it per routed row: routing_planner.hpp):
//
//   bits  | 31 ......... 24 | 23 .................. 8 | 7 ........... 0 |
//   field |     expert      |  token (on its home)    |    home chip    |
//
// Example: 0x05002A03 is expert 5's output for token 42 of chip 3. The row is summed on chip 3, on accumulator
// 42 % A (A accumulators a chip), as that accumulator's token 42 / A.
constexpr uint32_t header_home_chip(uint32_t header) { return header & 0xFFu; }
constexpr uint32_t header_token(uint32_t header) { return (header >> 8) & 0xFFFFu; }
constexpr uint32_t header_expert(uint32_t header) { return header >> 24; }

// The gate weight of a trailer word 1.
constexpr uint32_t trailer_weight_bits(uint32_t word) { return word & 0xFFFFu; }

// =====================================================================================================================
// Ring layout words (rings mode: one landing ring per source chip in each accumulator)
// =====================================================================================================================
//
// Each call an accumulator carves its landing pool into one ring per source chip (by that call's demand) and publishes
// each ring as one word; its home port forwards it to the source's port (LAYO -> LAYR in the state block):
//
//   bits  | 31 ...... 24 | 23 ........... 12 | 11 ........... 0 |
//   field |     tag      | first pool slot   |  depth (slots)  |
//
// tag is the call's number, 1..255 (never 0: a zeroed word is "no layout yet").
// Example: 0x01040008 is call 1's ring of 8 slots starting at pool slot 64. The source's u-th row this call goes to
// pool slot 64 + u % 8, once the ring has room.
constexpr uint32_t ring_layout(uint32_t tag, uint32_t first_slot, uint32_t depth) {
    return tag << 24 | first_slot << 12 | depth;
}
constexpr uint32_t ring_layout_tag(uint32_t word) { return word >> 24; }
constexpr uint32_t ring_layout_first_slot(uint32_t word) { return (word >> 12) & 0xFFFu; }
constexpr uint32_t ring_layout_depth(uint32_t word) { return word & 0xFFFu; }

// =====================================================================================================================
// Grant words (shared landing: slots granted to sources one at a time)
// =====================================================================================================================
//
// With the shared landing, an accumulator grants single pool slots to sources by id. Grant number q (counted per
// source and accumulator) is written into the home port's grant ring at q % G:
//
//   bits  | 31 ...... 24 | 23 .. 16 | 15 ................ 0 |
//   field | 0x80 | q%128 |    0     |   granted pool slot   |
//
// The tag's top bit makes it nonzero (a zeroed entry is "no grant"); its low 7 bits tell this grant from the one G
// entries earlier in the same ring entry.
// Example: 0x83000011 is grant 3 (or 131, ...) of pool slot 17.
constexpr uint32_t grant_tag(uint32_t q) { return 0x80u | (q & 0x7Fu); }
constexpr uint32_t grant_word(uint32_t q, uint32_t slot) { return grant_tag(q) << 24 | slot; }
constexpr uint32_t grant_word_tag(uint32_t word) { return word >> 24; }
constexpr uint32_t grant_word_slot(uint32_t word) { return word & 0xFFFFu; }

// =====================================================================================================================
// The MISC block of a combine core's state (StateLayout::MISC, kMiscWords uint32 words)
// =====================================================================================================================
//
// Scalars and small arrays of the core's kernels. Like the rest of the state block, it is zeroed at the start of
// every call (combine_zero_gate.hpp), so every counter here counts from 0 within one call.
//
//   word      | owner                          | contents
//   ----------+--------------------------------+--------------------------------------------------------------------
//   1         | port writer                    | credits returned: rows covered by the credit snapshots sent
//   101       | port (hand-off receiver)       | rows of the injection ring's hand-off before this call
//   103       | accumulator on a port's core   | rows of the local ring's hand-off before this call
//   104       | port writer                    | the call number (the ring layouts' tag)
//   105       | accumulator reader             | the call number (the ring layouts' tag)
//   128..159  | port (hand-off receiver)       | landed[w]: rows y writer w landed in the injection ring
//   160..191  | accumulator on a port's core   | landed[w]: rows y writer w landed in the local ring
//   208..223  | port writer                    | scratch: the device plan's n_inj[8], n_ret[8] (64 B aligned)
//   224..239  | port writer                    | scratch: the header cache's key block (64 B)
//   256..287  | port writer                    | rows consumed per source chip (accumulators increment)
//   288..319  | port writer                    | rows returned per source chip (covered by the snapshots sent)
//   320..351  | accumulator reader             | rows taken per ring (source chip)
//
// An accumulator on its own core uses words 101 and 128.. for its local ring (no port shares the core); one on a
// port's core uses 103 and 160.. (the port has 101 and 128..). The y writers find the landed arrays through their
// runtime args (flat_program.hpp kFlatYL1Args).
inline constexpr uint32_t kMiscWords = 512;
inline constexpr uint32_t kMiscBytes = 4 * kMiscWords;
inline constexpr uint32_t kMiscReturned = 1;
inline constexpr uint32_t kMiscPortHandoffBase = 101;
inline constexpr uint32_t kMiscAccHandoffBaseOnPort = 103;
inline constexpr uint32_t kMiscPortCalls = 104;
inline constexpr uint32_t kMiscAccCalls = 105;
inline constexpr uint32_t kMiscPortLanded = 128;
inline constexpr uint32_t kMiscAccLandedOnPort = 160;
inline constexpr uint32_t kMiscPlanScratch = 208;
inline constexpr uint32_t kMiscHeaderCacheKey = 224;
inline constexpr uint32_t kMiscRowsConsumed = 256;
inline constexpr uint32_t kMiscRowsReturned = 288;
inline constexpr uint32_t kMiscTaken = 320;
inline constexpr uint32_t kMaxWriters = 32;  // y writers at most (a landed[] array)
inline constexpr uint32_t kMaxChips = 32;    // chips at most (a per-source array)

// An accumulator's hand-off words: on its own core, or on a port's core.
constexpr uint32_t acc_handoff_base_word(bool on_port) {
    return on_port ? kMiscAccHandoffBaseOnPort : kMiscPortHandoffBase;
}
constexpr uint32_t acc_landed_word(bool on_port) { return on_port ? kMiscAccLandedOnPort : kMiscPortLanded; }

static_assert(kMiscPortLanded + kMaxWriters <= kMiscAccLandedOnPort, "landed arrays overlap");
static_assert(kMiscAccLandedOnPort + kMaxWriters <= kMiscPlanScratch, "landed array overlaps the plan scratch");
static_assert(kMiscPlanScratch % 16 == 0 && kMiscPlanScratch + 16 <= kMiscHeaderCacheKey, "plan scratch: 64 B");
static_assert(kMiscHeaderCacheKey % 16 == 0 && kMiscHeaderCacheKey + 16 <= kMiscRowsConsumed, "key block: 64 B");
static_assert(kMiscRowsConsumed + kMaxChips <= kMiscRowsReturned, "per-source arrays overlap");
static_assert(kMiscRowsReturned + kMaxChips <= kMiscTaken, "per-source arrays overlap");
static_assert(kMiscTaken + kMaxChips <= kMiscWords, "per-ring array past the block");

}  // namespace combine_formats
