// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "routing_metadata_faces.hpp"

namespace routing_metadata {

constexpr uint32_t kFacePairWords = 2 * kFaceWords;

// The scores' 16-column groups that hold num_experts columns.
constexpr uint32_t score_groups(uint32_t num_experts) { return (num_experts + kWordsPerToken - 1) / kWordsPerToken; }

// Issues the reads of the first `groups` 16-column groups of the (1, 1, tokens, C) scores' tile rows
// [first_tile_row, first_tile_row + num_tile_rows). Group g of a tile row is its tile g / 2, faces
// g % 2 (rows 0-15) and g % 2 + 2 (rows 16-31); it lands as face pair tile_row * groups + g from
// destination, so token i's column 16 g + c is at word 16 i + c of that pair. The caller waits for
// them with async_read_barrier.
template <typename Accessor>
void read_score_groups(
    Noc& noc,
    const Accessor& scores,
    uint32_t destination,
    uint32_t first_tile_row,
    uint32_t num_tile_rows,
    uint32_t groups,
    uint32_t tiles_per_row) {
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t pair = destination + (tile_row * groups + group) * kFacePairBytes;
            const uint32_t page = (first_tile_row + tile_row) * tiles_per_row + group / 2;
            const uint32_t face = (group % 2) * kFaceBytes;
            noc.async_read(
                scores, CoreLocalMem<uint32_t>(pair), kFaceBytes, {.page_id = page, .offset_bytes = face}, {});
            noc.async_read(
                scores,
                CoreLocalMem<uint32_t>(pair + kFaceBytes),
                kFaceBytes,
                {.page_id = page, .offset_bytes = face + kFaceTwoOffset},
                {});
        }
    }
}

// float32 bits to a key whose unsigned order is the numeric order, -0.0 equal to 0.0. A score of
// at least 0 keeps its bits as its key, which orders the same.
inline uint32_t order_key(uint32_t bits) {
    bits = bits == 0x80000000u ? 0u : bits;
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

// ids gets the top_k of keys, largest first and a tie to the lower index.
template <uint32_t num_experts, uint32_t top_k>
inline void top_k_of(const uint32_t (&keys)[num_experts], uint32_t (&ids)[top_k]) {
    if constexpr (top_k == 2) {
        // Two compares a key: most fail the first.
        uint32_t first = keys[1] > keys[0] ? 1 : 0;
        uint32_t second = 1 - first;
        uint32_t first_key = keys[first];
        uint32_t second_key = keys[second];
#pragma GCC unroll 64
        for (uint32_t expert = 2; expert < num_experts; ++expert) {
            const uint32_t key = keys[expert];
            if (key > second_key) {
                if (key > first_key) {
                    second = first;
                    second_key = first_key;
                    first = expert;
                    first_key = key;
                } else {
                    second = expert;
                    second_key = key;
                }
            }
        }
        ids[0] = first;
        ids[1] = second;
    } else {
        uint32_t kept_keys[top_k];
        uint32_t kept = 0;
        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            const uint32_t key = keys[expert];
            uint32_t slot;
            if (kept < top_k) {
                slot = kept++;
            } else if (key > kept_keys[top_k - 1]) {
                slot = top_k - 1;
            } else {
                continue;
            }
            // Moves past the smaller kept keys only: an equal key keeps its lower index ahead.
            for (; slot > 0 && kept_keys[slot - 1] < key; --slot) {
                kept_keys[slot] = kept_keys[slot - 1];
                ids[slot] = ids[slot - 1];
            }
            kept_keys[slot] = key;
            ids[slot] = expert;
        }
    }
}

// For each token of the num_tile_rows tile rows read_score_groups put at faces, selects its top_k of
// the first num_experts columns, largest first and a tie to the lower column, the order
// ttnn.topk(stable=True) gives. Choice k of token i of tile row r goes to word r * 512 + 16 i + k of
// faces, the layout read_face_pairs gives a selection: the expert's id, or with write_ids false its
// score's bits. Tile row r's scores sit at or past the face pair its choices take, and the rows run
// in order, so a choice only ever overwrites scores already read.
template <uint32_t num_experts, uint32_t top_k, bool write_ids>
void select_top_k(volatile tt_l1_ptr uint32_t* faces, uint32_t num_tile_rows) {
    constexpr uint32_t groups = score_groups(num_experts);
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        const volatile tt_l1_ptr uint32_t* scores = faces + tile_row * groups * kFacePairWords;
        volatile tt_l1_ptr uint32_t* choices = faces + tile_row * kFacePairWords;
        for (uint32_t token = 0; token < kBlockRows; ++token) {
            // Loaded before any is compared, so the L1 load latencies overlap.
            uint32_t bits[num_experts];
#pragma GCC unroll 64
            for (uint32_t expert = 0; expert < num_experts; ++expert) {
                bits[expert] = scores
                    [(expert / kWordsPerToken) * kFacePairWords + token * kWordsPerToken + expert % kWordsPerToken];
            }
            uint32_t signs = 0;
#pragma GCC unroll 64
            for (uint32_t expert = 0; expert < num_experts; ++expert) {
                signs |= bits[expert];
            }
            uint32_t keys[num_experts];
#pragma GCC unroll 64
            for (uint32_t expert = 0; expert < num_experts; ++expert) {
                keys[expert] = (signs & 0x80000000u) != 0 ? order_key(bits[expert]) : bits[expert];
            }
            uint32_t ids[top_k];
            top_k_of<num_experts, top_k>(keys, ids);
#pragma GCC unroll 16
            for (uint32_t k = 0; k < top_k; ++k) {
                choices[token * kWordsPerToken + k] = write_ids ? ids[k] : bits[ids[k]];
            }
        }
    }
}

}  // namespace routing_metadata
