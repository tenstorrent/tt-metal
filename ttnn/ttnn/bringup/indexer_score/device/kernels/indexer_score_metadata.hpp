// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

struct IndexerScoreMetadataBounds {
    uint32_t kv_len_tiles;
    uint32_t chunk_start_tiles;
    uint32_t straddle_q_tile;
    uint32_t straddle_jump_tiles;
};

inline constexpr uint32_t indexer_score_kv_len_tiles_word =
    offsetof(IndexerScoreMetadataBounds, kv_len_tiles) / sizeof(uint32_t);
inline constexpr uint32_t indexer_score_chunk_start_tiles_word =
    offsetof(IndexerScoreMetadataBounds, chunk_start_tiles) / sizeof(uint32_t);
inline constexpr uint32_t indexer_score_straddle_q_tile_word =
    offsetof(IndexerScoreMetadataBounds, straddle_q_tile) / sizeof(uint32_t);
inline constexpr uint32_t indexer_score_straddle_jump_tiles_word =
    offsetof(IndexerScoreMetadataBounds, straddle_jump_tiles) / sizeof(uint32_t);

static_assert(sizeof(IndexerScoreMetadataBounds) == 4 * sizeof(uint32_t));
