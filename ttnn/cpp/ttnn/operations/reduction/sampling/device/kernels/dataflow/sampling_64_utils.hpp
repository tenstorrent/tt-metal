// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace sampling_64 {

// Element offset, independent of the value/index element byte width. Candidate
// columns 32..63 live in a second complete tile, not the next user's face row.
constexpr uint32_t candidate_offset(uint32_t user, uint32_t rank) {
    return (rank / 32) * 1024 + (user / 16) * 512 + (user % 16) * 16 + ((rank % 32) / 16) * 256 + rank % 16;
}

// The caller supplies BF16 probabilities widened exactly to float. Preserve the
// legacy writer's strict cumulative > cutoff, FP32 accumulation, and first-rank
// fallback. The default 32 path does not call this helper.
template <typename ProbabilityAt>
uint32_t sample_rank(uint32_t k, float p, float threshold, ProbabilityAt probability_at) {
    float mass = 0.0f;
    uint32_t kept = 0;
    for (; kept < k; ++kept) {
        mass += probability_at(kept);
        if (mass > p) {
            ++kept;
            break;
        }
    }
    float cumulative = 0.0f;
    for (uint32_t rank = 0; rank < kept; ++rank) {
        cumulative += probability_at(rank) / mass;
        if (cumulative > threshold) {
            return rank;
        }
    }
    return 0;
}

}  // namespace sampling_64
