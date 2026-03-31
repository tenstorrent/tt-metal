// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>
#include <array>

//=============================================================================
// MoEGPT Ring All-to-All Configuration for GPT-OSS
//
// Dimensions:
//   hidden_size (K) = 2880 -> 90 tiles
//   intermediate_size (N) = 2880 -> 90 tiles
//   experts/device = 4
//
// W0/W1: [K, N] = [90, 90] tiles -> 7 or 8 tiles/core (90/12 = 7.5)
// W2:    [N, K] = [90, 90] tiles -> 7 or 8 tiles/core (90/12 = 7.5)
//
// Both W0/W1 and W2 have the same distribution since K == N == 2880.
// Boundary-optimized: cores {0,1,4,5,8,9} get 8 tiles, rest get 7.
//=============================================================================

namespace moe_gpt_ring {

constexpr uint32_t NUM_CORES = 12;

// W0/W1 weight height in tiles: K / 32 = 2880 / 32 = 90
constexpr uint32_t NUM_W0_W1_TILES_H = 90;
constexpr uint32_t NUM_W0_W1_TILES_PLUS_BIAS_H = 91;

// W2 weight height in tiles: N / 32 = 2880 / 32 = 90
constexpr uint32_t NUM_W2_TILES_H = 90;
constexpr uint32_t NUM_W2_TILES_PLUS_BIAS_H = 91;

// Transaction sizing for DRAM reads
// Each transaction = 14 tiles of Bfp4_b = 14 * 576 = 8064 bytes (fits in 8KB NOC packet)
constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
constexpr uint32_t W0_W1_TILES_PER_TXN = 14;

constexpr uint32_t W2_TXNS_PER_BLOCK = 2;
constexpr uint32_t W2_TILES_PER_TXN = 14;

constexpr uint32_t W2_B2_BLOCKS_PER_EXPERT =
    (((NUM_W2_TILES_PLUS_BIAS_H * 8) - 1) / (W2_TILES_PER_TXN * W2_TXNS_PER_BLOCK)) + 1;
constexpr uint32_t W0_B0_W1_B1_BLOCKS_PER_EXPERT = W2_B2_BLOCKS_PER_EXPERT * 2;

//-----------------------------------------------------------------------------
// Precomputed lookup tables
// W0/W1 width = 90 tiles / 12 cores = 7.5 -> 6 cores get 8, 6 cores get 7
// Source tiles: {8,8,7,7,8,8,7,7,8,8,7,7}
// Each row[core][step] = source_tiles[(core - step) mod 12]
//-----------------------------------------------------------------------------

// Boundary-optimized: cores {0,1,4,5,8,9} get 8 tiles
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP_A[NUM_CORES][NUM_CORES] = {
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
// Boundary-optimized: pairs at [0,1], [4,5], [8,9] get 8 tiles
constexpr uint32_t W2_TILES_PER_CORE_A[NUM_CORES] = {
    8,
    8,
    7,
    7,
    8,
    8,
    7,
    7,
    8,
    8,
    7,
    7,
};

// IN2_TILES_PER_STEP = max of W0_W1 tiles per core per step = 8
constexpr uint32_t IN2_TILES_PER_STEP_A = *std::max_element(
    W0_W1_TILES_PER_CORE_PER_STEP_A[0], W0_W1_TILES_PER_CORE_PER_STEP_A[0] + NUM_CORES, [](uint32_t a, uint32_t b) {
        return a < b;
    });

// NUM_A2A_ITERS = max(W2_TILES_PER_CORE) / 4 = 8 / 4 = 2
constexpr uint32_t NUM_A2A_ITERS_A = *std::max_element(
                                         W2_TILES_PER_CORE_A,
                                         W2_TILES_PER_CORE_A + NUM_CORES,
                                         [](uint32_t a, uint32_t b) { return (a / 2) < (b / 2); }) /
                                     4;

// Tokens per chunk (1 tile height)
constexpr uint32_t TOKENS_PER_CHUNK = 32;

// Combine output grid: token-parallel rows × data-parallel columns.
// Each core's buffer packs all experts sequentially for its token slice.
constexpr uint32_t COMBINE_WIDTH_SHARD_DIM = 3;   // data-parallel (hidden dim split)
constexpr uint32_t COMBINE_HEIGHT_SHARD_DIM = 4;  // token-parallel (token dim split)
constexpr uint32_t K_TILES = NUM_W0_W1_TILES_H;   // 90

// Ring cores per combine column: 12 / 3 = 4
constexpr uint32_t RING_CORES_PER_COMBINE_COL = NUM_CORES / COMBINE_WIDTH_SHARD_DIM;

// Combine shard width in tiles: 90 / 3 = 30
constexpr uint32_t COMBINE_SHARD_WIDTH_TILES = K_TILES / COMBINE_WIDTH_SHARD_DIM;

// Source width tiles (max tiles per core for untilize page sizing)
constexpr uint32_t SOURCE_WIDTH_TILES = IN2_TILES_PER_STEP_A;  // 8

// Precomputed combine width offset per ring core (for output shard placement)
constexpr std::array<uint32_t, NUM_CORES> COMBINE_W_OFFSET_PER_CORE_A = []() constexpr {
    std::array<uint32_t, NUM_CORES> arr = {};
    uint32_t sum = 0;
    for (uint32_t i = 0; i < NUM_CORES; ++i) {
        arr[i] = sum;
        sum += W2_TILES_PER_CORE_A[i];
    }
    return arr;
}();

}  // namespace moe_gpt_ring
