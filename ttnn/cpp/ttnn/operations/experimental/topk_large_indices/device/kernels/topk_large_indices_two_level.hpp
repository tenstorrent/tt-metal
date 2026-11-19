// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by the three kernels of a two-level program. A two-level program selects each row's top k from the
// maxima of 32-key blocks first, then from the keys of the 2048 selected blocks only:
//   stage 1: block maxima of the valid prefix (compute: tilize + max reduce; writer: dense maxima row);
//   stage 2: top-2048 of the maxima -> selected block ids (compute), ascending block table (writer);
//   stage 3: the selected blocks gathered into one 65,536-key candidate row (reader), its exact top-2048
//            (compute), candidate positions mapped back to keys (writer).
// The program decides once, from the search length, whether its rows run this way or the single-level way. The
// two-level code compiles only with TOPK_TWO_LEVEL defined, so single-level programs build exactly as before.

#include <cstdint>

namespace topk_two_level {

constexpr uint32_t block_keys = 32;
constexpr uint32_t block_bytes = block_keys * sizeof(uint16_t);
constexpr uint32_t selected_blocks = 2048;
constexpr uint32_t candidate_keys = selected_blocks * block_keys;
// Keys per stage-1 tile: a 32x32 tile whose rows are 32 consecutive keys.
constexpr uint32_t tile_keys = 1024;
constexpr uint32_t sentinel = 0xFFFFFFFFu;

// The reader-to-compute mailbox of a two-level program: the single-level bounds (whole row, split half), then
// the mode and the search length.
constexpr uint32_t mailbox_mode_word = 4;
constexpr uint32_t mailbox_search_length_word = 5;
// The reader-to-writer mailbox page.
constexpr uint32_t writer_mailbox_mode_word = 0;
constexpr uint32_t writer_mailbox_search_length_word = 1;

constexpr uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Whether a program runs two-level for this search length. An empty row (a real-token end of 0) runs two-level
// too: with no blocks every stage passes -inf through, so the row comes out as sentinels.
constexpr bool two_level_mode_for(uint32_t search_len, uint32_t cutover) {
    return search_len >= cutover || search_len == 0;
}

// Stage-2 chunks of 2048 maxima for a search length: at least two, so every stage runs on the split body. Maxima
// past the row's blocks are -inf.
constexpr uint32_t stage2_chunks(uint32_t search_len) {
    const uint32_t chunks = div_up(div_up(search_len, block_keys), 2048);
    return chunks < 2 ? 2 : chunks;
}

}  // namespace topk_two_level
