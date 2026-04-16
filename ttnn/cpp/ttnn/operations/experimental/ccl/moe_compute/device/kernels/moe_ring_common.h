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

constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
constexpr uint32_t W0_W1_TILES_PER_TXN = 14;

constexpr uint32_t W2_TXNS_PER_BLOCK = 2;
constexpr uint32_t W2_TILES_PER_TXN = 14;

constexpr uint32_t TOKENS_PER_CHUNK = 32;

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
    static constexpr uint32_t NUM_W0_W1_TILES_H = 224;
    static constexpr uint32_t NUM_W2_TILES_H = 64;

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

// GPT configuration for hidden_size = intermediate_size = 2880 (90 tiles)
// Note: This config assumes NUM_W0_W1_TILES_H = NUM_W2_TILES_H = 90
// For GPT-OSS: K = N = 2880, so both W0/W1 and W2 have 90x90 tile dimensions
struct GptRingConfig {
    // GPT-specific tile heights (vs 224/64 in DeepSeek)
    // static constexpr uint32_t GPT_W0_W1_TILES_H = 90;
    // static constexpr uint32_t GPT_W2_TILES_H = 90;

    // static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
    //     (((91 * 8) - 1) / (W2_TILES_PER_TXN * W2_TXNS_PER_BLOCK)) + 1;  // Different calculation for GPT

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT;

    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 3;  // vs 4 in DeepSeek
    static constexpr uint32_t OUTPUT_HEIGHT_SHARD_DIM = 4;

    // Boundary-optimized: tiles[core_id][step] - cores {0,1,4,5,8,9} get 8 tiles
    static constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[NUM_CORES][NUM_CORES] = {
        // Core 0: sources [0,11,10,9,8,7,6,5,4,3,2,1]
        {8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8},
        // Core 1: sources [1,0,11,10,9,8,7,6,5,4,3,2]
        {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7},
        // Core 2: sources [2,1,0,11,10,9,8,7,6,5,4,3]
        {7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7},
        // Core 3: sources [3,2,1,0,11,10,9,8,7,6,5,4]
        {7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8},
        // Core 4: sources [4,3,2,1,0,11,10,9,8,7,6,5]
        {8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8},
        // Core 5: sources [5,4,3,2,1,0,11,10,9,8,7,6]
        {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7},
        // Core 6: sources [6,5,4,3,2,1,0,11,10,9,8,7]
        {7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7},
        // Core 7: sources [7,6,5,4,3,2,1,0,11,10,9,8]
        {7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8},
        // Core 8: sources [8,7,6,5,4,3,2,1,0,11,10,9]
        {8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8},
        // Core 9: sources [9,8,7,6,5,4,3,2,1,0,11,10]
        {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7},
        // Core 10: sources [10,9,8,7,6,5,4,3,2,1,0,11]
        {7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7},
        // Core 11: sources [11,10,9,8,7,6,5,4,3,2,1,0]
        {7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8},
    };

    // W2 tiles per core: 90 tiles / 12 cores = 7.5 -> 6 cores get 8, 6 cores get 7
    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7};

    static constexpr auto IN2_TILES_PER_STEP = detail::compute_in2_tiles_per_step(W0_W1_TILES_PER_CORE_PER_STEP[0]);

    // GPT uses a different algorithm with custom comparator
    static constexpr auto NUM_A2A_ITERS = detail::compute_a2a_iters(W2_TILES_PER_CORE, OUTPUT_WIDTH_SHARD_DIM);

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core(W2_TILES_PER_CORE);

    // Additional GPT-specific constants
    static constexpr uint32_t COMBINE_SHARD_WIDTH_TILES = 90 / OUTPUT_WIDTH_SHARD_DIM;          // 30
    static constexpr uint32_t RING_CORES_PER_COMBINE_COL = NUM_CORES / OUTPUT_WIDTH_SHARD_DIM;  // 4
};

}  // namespace moe_ring
