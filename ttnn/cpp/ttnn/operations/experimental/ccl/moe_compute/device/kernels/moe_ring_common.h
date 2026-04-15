// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <algorithm>
#include <stdint.h>

namespace detail {

enum class MoEActivationFunction : uint8_t { SILU = 0, SWIGLU = 1 };

}  // namespace detail
//=============================================================================
// MoE Ring All-to-All Configuration
//
// Two arrangements supported:
// 1. Boundary-optimized: [6,6,5,5,5,5,5,5,6,6,5,5] - positions [0,1,8,9]
//    Formula: ((source & 7) < 2) ? 6 : 5
//    Best for dm0 block boundary alignment (9 boundary waits)
//
// 2. Evenly distributed: [6,5,5,6,5,5,6,5,5,6,5,5] - positions [0,3,6,9]
//    Formula: (source % 3 == 0) ? 6 : 5
//    Simplest code pattern
//=============================================================================

namespace moe_ring {

constexpr uint32_t NUM_CORES = 12;

constexpr uint32_t NUM_W0_W1_TILES_H = 224;
constexpr uint32_t NUM_W2_TILES_H = 64;

constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
constexpr uint32_t W0_W1_TILES_PER_TXN = 14;

constexpr uint32_t W2_TXNS_PER_BLOCK = 2;
constexpr uint32_t W2_TILES_PER_TXN = 14;

//-----------------------------------------------------------------------------
// Precomputed lookup tables (generated at compile time)
// Use these if you prefer lookup over runtime computation
//-----------------------------------------------------------------------------

namespace detail {

constexpr uint32_t compute_a2a_iters(const uint32_t* arr, const uint32_t width_shard_dim) {
    return (*std::max_element(arr, arr + NUM_CORES) + width_shard_dim - 1) / width_shard_dim;
};

constexpr uint32_t compute_in2_tiles_per_step(const uint32_t* arr) {
    return *std::max_element(arr, arr + NUM_CORES, [](uint32_t a, uint32_t b) { return a < b; });
};

constexpr std::array<uint32_t, NUM_CORES> compute_combine_w_offset_per_core(const uint32_t* arr) {
    std::array<uint32_t, NUM_CORES> out = {};
    uint32_t sum = 0;
    for (uint32_t i = 0; i < NUM_CORES; ++i) {
        out[i] = sum;
        sum += arr[i];
    }
    return out;
}

}  // namespace detail

struct DeepSeekRingConfig {
    static constexpr uint32_t W2_BLOCKS_PER_EXPERT = 50;
    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 4;

    // Evenly distributed: tiles[core_id][step]
    static constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[NUM_CORES][NUM_CORES] = {
        // Core 0: pattern [6,5,5,6,5,5,6,5,5,6,5,5]
        {6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5},
        // Core 1: pattern [5,6,5,5,6,5,5,6,5,5,6,5]
        {5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5},
        // Core 2: pattern [5,5,6,5,5,6,5,5,6,5,5,6]
        {5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6},
        // Core 3: same as Core 0
        {6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5},
        // Core 4: same as Core 1
        {5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5},
        // Core 5: same as Core 2
        {5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6},
        // Core 6: same as Core 0
        {6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5},
        // Core 7: same as Core 1
        {5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5},
        // Core 8: same as Core 2
        {5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6},
        // Core 9: same as Core 0
        {6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5},
        // Core 10: same as Core 1
        {5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6, 5},
        // Core 11: same as Core 2
        {5, 5, 6, 5, 5, 6, 5, 5, 6, 5, 5, 6},
    };

    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {
        18,
        19,
        19,
        18,
        19,
        19,
        18,
        19,
        19,
        18,
        19,
        19,
    };

    static constexpr auto IN2_TILES_PER_STEP = detail::compute_in2_tiles_per_step(W0_W1_TILES_PER_CORE_PER_STEP[0]);

    static constexpr auto NUM_A2A_ITERS = detail::compute_a2a_iters(W2_TILES_PER_CORE, OUTPUT_WIDTH_SHARD_DIM);

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core(W2_TILES_PER_CORE);
};

// Create aliases at namespace level for backward compatibility
constexpr auto W2_BLOCKS_PER_EXPERT = DeepSeekRingConfig::W2_BLOCKS_PER_EXPERT;
constexpr auto& W0_W1_TILES_PER_CORE_PER_STEP = DeepSeekRingConfig::W0_W1_TILES_PER_CORE_PER_STEP;
constexpr auto& W2_TILES_PER_CORE = DeepSeekRingConfig::W2_TILES_PER_CORE;
constexpr auto IN2_TILES_PER_STEP = DeepSeekRingConfig::IN2_TILES_PER_STEP;
constexpr auto NUM_A2A_ITERS = DeepSeekRingConfig::NUM_A2A_ITERS;
constexpr auto& COMBINE_W_OFFSET_PER_CORE = DeepSeekRingConfig::COMBINE_W_OFFSET_PER_CORE;

}  // namespace moe_ring
