// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The schedule page: how a data-movement kernel hands the call's expert schedule (expert_schedule.hpp) to the compute
// kernel on its core. One page of CB 6, uint32 words; the compute reads them with read_tile_value.
//
//   word                      | contents
//   --------------------------+----------------------------------------------------------------------------------------
//   kEntries                  | N: the schedule's entries
//   kTotalSubBlocks           | sub-blocks over all entries
//   kSubBlocks + a      (a<N) | entry a's sub-blocks
//   kLoadWord + a       (a<N) | entry a's gate/up weight load, as a load word (below)
//   kLastRowTiles + a   (a<N) | row tiles (32 rows) holding tokens in entry a's last sub-block (the rest: skipped)
//
// A load word:
//
//   bits  | 31 ...... 17 | 16       | 15 ........ 8 | 7 ......... 0 |
//   field |      0       | last use | ring region   | load          |
//
// load: the load's place in the gate/up weight stream; ring region: where in the weight ring the load sits; last use:
// the load's weights are freed after this entry.
//
// Example: two entries, 300 and 40 tokens, 128-row sub-blocks, each entry its own load:
//   kEntries 2, kTotalSubBlocks 4, sub-blocks [3, 1], load words [0x10000, 0x10101], last row tiles [2, 2]
// (300 tokens: sub-blocks of 128, 128 and 44 rows, the last two row tiles; 40 tokens: one sub-block, two row tiles)
#pragma once

#include <cstdint>

#include "flat_options.hpp"

namespace schedule_page {

// schedule entries at most: a pinned expert's chunks plus the other experts (expert_schedule.hpp)
constexpr uint32_t kMaxEntries = 2 * flat_options::max_experts;
static_assert(kMaxEntries < 255, "the schedule keeps entry and load indices in 8 bits (0xFF: none)");

constexpr uint32_t kEntries = 0;
constexpr uint32_t kTotalSubBlocks = 1;
constexpr uint32_t kSubBlocks = 2;
constexpr uint32_t kLoadWord = kSubBlocks + kMaxEntries;
constexpr uint32_t kLastRowTiles = kLoadWord + kMaxEntries;
constexpr uint32_t kWords = kLastRowTiles + kMaxEntries;  // the page holds this many words (host: schedule_page_bytes)

constexpr uint32_t load_word(uint32_t load, uint32_t ring_region, bool last_use) {
    return load | (ring_region << 8) | (static_cast<uint32_t>(last_use) << 16);
}
constexpr uint32_t load_word_load(uint32_t word) { return word & 0xFFu; }
constexpr uint32_t load_word_ring_region(uint32_t word) { return (word >> 8) & 0xFFu; }
constexpr bool load_word_last_use(uint32_t word) { return ((word >> 16) & 1u) != 0; }

}  // namespace schedule_page
