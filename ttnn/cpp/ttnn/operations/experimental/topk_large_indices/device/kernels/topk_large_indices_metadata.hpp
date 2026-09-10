// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

struct TopkMetadataBounds {
    uint32_t num_chunks;
    uint32_t tail_elements;
};

constexpr TopkMetadataBounds calculate_topk_bounds(uint32_t search_len, uint32_t chunk_elements) {
    const uint32_t num_chunks = (search_len + chunk_elements - 1) / chunk_elements;
    return {
        .num_chunks = num_chunks,
        .tail_elements = search_len - ((num_chunks - 1) * chunk_elements),
    };
}

inline constexpr uint32_t topk_metadata_num_chunks_word = offsetof(TopkMetadataBounds, num_chunks) / sizeof(uint32_t);
inline constexpr uint32_t topk_metadata_tail_elements_word =
    offsetof(TopkMetadataBounds, tail_elements) / sizeof(uint32_t);

static_assert(sizeof(TopkMetadataBounds) == 2 * sizeof(uint32_t));
