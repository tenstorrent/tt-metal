// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>

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

constexpr uint32_t W2_BLOCKS_PER_EXPERT = 50;

//-----------------------------------------------------------------------------
// Precomputed lookup tables (generated at compile time)
// Use these if you prefer lookup over runtime computation
//-----------------------------------------------------------------------------

// Boundary-optimized: tiles[core_id][step]
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP_A[NUM_CORES][NUM_CORES] = {
    // Core 0: sources [0,11,10,9,8,7,6,5,4,3,2,1]
    {6, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 6},
    // Core 1: sources [1,0,11,10,9,8,7,6,5,4,3,2]
    {6, 6, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5},
    // Core 2: sources [2,1,0,11,10,9,8,7,6,5,4,3]
    {5, 6, 6, 5, 5, 6, 6, 5, 5, 5, 5, 5},
    // Core 3: sources [3,2,1,0,11,10,9,8,7,6,5,4]
    {5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 5, 5},
    // Core 4: sources [4,3,2,1,0,11,10,9,8,7,6,5]
    {5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 5},
    // Core 5: sources [5,4,3,2,1,0,11,10,9,8,7,6]
    {5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5, 5},
    // Core 6: sources [6,5,4,3,2,1,0,11,10,9,8,7]
    {5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6, 5},
    // Core 7: sources [7,6,5,4,3,2,1,0,11,10,9,8]
    {5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6, 6},
    // Core 8: sources [8,7,6,5,4,3,2,1,0,11,10,9]
    {6, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6},
    // Core 9: sources [9,8,7,6,5,4,3,2,1,0,11,10]
    {6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 5, 5},
    // Core 10: sources [10,9,8,7,6,5,4,3,2,1,0,11]
    {5, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 5},
    // Core 11: sources [11,10,9,8,7,6,5,4,3,2,1,0]
    {5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6},
};

// Evenly distributed: tiles[core_id][step]
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP_B[NUM_CORES][NUM_CORES] = {
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

constexpr uint32_t W2_TILES_PER_CORE_A[NUM_CORES] = {
    18,
    18,
    19,
    19,
    19,
    19,
    19,
    19,
    18,
    18,
    19,
    19,
};

constexpr uint32_t W2_TILES_PER_CORE_B[NUM_CORES] = {
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

constexpr uint32_t IN2_TILES_PER_STEP_A = *std::max_element(
    W0_W1_TILES_PER_CORE_PER_STEP_A[0], W0_W1_TILES_PER_CORE_PER_STEP_A[0] + NUM_CORES, [](uint32_t a, uint32_t b) {
        return a < b;
    });

constexpr uint32_t IN2_TILES_PER_STEP_B = *std::max_element(
    W0_W1_TILES_PER_CORE_PER_STEP_B[0], W0_W1_TILES_PER_CORE_PER_STEP_B[0] + NUM_CORES, [](uint32_t a, uint32_t b) {
        return a < b;
    });

constexpr uint32_t NUM_A2A_ITERS_A = *std::max_element(
                                         W2_TILES_PER_CORE_A,
                                         W2_TILES_PER_CORE_A + NUM_CORES,
                                         [](uint32_t a, uint32_t b) { return (a / 2) < (b / 2); }) /
                                     4;

constexpr uint32_t NUM_A2A_ITERS_B = *std::max_element(
                                         W2_TILES_PER_CORE_B,
                                         W2_TILES_PER_CORE_B + NUM_CORES,
                                         [](uint32_t a, uint32_t b) { return (a / 2) < (b / 2); }) /
                                     4;

constexpr std::array<uint32_t, NUM_CORES> COMBINE_W_OFFSET_PER_CORE_B = []() constexpr {
    std::array<uint32_t, NUM_CORES> arr = {};
    uint32_t sum = 0;
    for (uint32_t i = 0; i < NUM_CORES; ++i) {
        arr[i] = sum;
        sum += W2_TILES_PER_CORE_B[i];
    }
    return arr;
}();

}  // namespace moe_ring
