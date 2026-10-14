// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "chunked_q_mapping.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

struct StripedHaloRange {
    uint32_t begin = 0;
    uint32_t end = 0;
    constexpr bool operator==(const StripedHaloRange& other) const { return begin == other.begin && end == other.end; }
};

// Runtime block shared by host, reader and writer. Only prefix/end change on cache hits.
// Row quantities are tiles; prefix, logical end and window are tokens.
struct StripedHaloGeometry {
    uint32_t prefix_tokens;
    uint32_t logical_tokens;
    uint32_t q_local_rows;
    uint32_t cache_local_rows;
    uint32_t stripe_rows;
    uint32_t k_chunk_rows;
    uint32_t window_tokens;
    uint32_t ring_size;
    uint32_t source_rank;
    uint32_t peer_count;
    uint32_t backward;
    uint32_t num_links;

    constexpr uint32_t peer(uint32_t index) const {
        return backward ? source_rank - index - 1 : source_rank + index + 1;
    }

    constexpr StripedHaloRange range(uint32_t index) const {
        const auto mapping = build_chunked_q_mapping(
            prefix_tokens / 32, logical_tokens / 32, q_local_rows, ring_size, peer(index), stripe_rows);
        if (mapping.q_valid_tile_count == 0) {
            return {};
        }
        const uint32_t first = striped_q_global_tile(mapping.q_pre_wrap_start_tile, 0, stripe_rows, ring_size);
        const uint32_t last =
            striped_q_global_tile(
                mapping.q_pre_wrap_start_tile, mapping.q_valid_tile_count - 1, stripe_rows, ring_size) +
            1;
        const uint32_t left = (window_tokens - 1 + 31) / 32;
        const uint32_t begin =
            striped_local_prefix(first > left ? first - left : 0, stripe_rows, ring_size, source_rank);
        const uint32_t end = striped_local_prefix(last, stripe_rows, ring_size, source_rank);
        if (begin >= end) {
            return {};
        }
        const uint32_t rounded_end = (end + k_chunk_rows - 1) / k_chunk_rows * k_chunk_rows;
        return {begin / k_chunk_rows * k_chunk_rows, rounded_end < cache_local_rows ? rounded_end : cache_local_rows};
    }

    constexpr uint32_t run_end(uint32_t begin, const StripedHaloRange& current) const {
        uint32_t end = begin + 1;
        while (end < peer_count && range(end) == current) {
            ++end;
        }
        return end;
    }
};

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
