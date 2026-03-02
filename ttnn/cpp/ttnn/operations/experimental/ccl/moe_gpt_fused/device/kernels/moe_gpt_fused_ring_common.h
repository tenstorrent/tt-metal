// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>

//=============================================================================
// MoEGPT Fused Ring All-to-All Configuration for GPT-OSS
//
// Same ring topology as moe_gpt, but with dynamic per-expert token counts
// received from gather cores via metadata CB.
//
// Dimensions:
//   hidden_size (K) = 2880 -> 90 tiles
//   intermediate_size (N) = 2880 -> 90 tiles
//   experts/device = 4
//   tokens_per_chunk = 32 -> 1 tile height
//
// W0/W1: [K, N] = [90, 90] tiles -> 7 or 8 tiles/core (90/12 = 7.5)
// W2:    [N, K] = [90, 90] tiles -> 7 or 8 tiles/core (90/12 = 7.5)
//=============================================================================

namespace moe_gpt_fused_ring {

constexpr uint32_t NUM_CORES = 12;
constexpr uint32_t NUM_GATHER_CORES = 3;
constexpr uint32_t NUM_COMBINE_CORES = 12;
constexpr uint32_t COMBINE_WIDTH_SHARD_DIM = 3;
constexpr uint32_t COMBINE_HEIGHT_SHARD_DIM = 4;

// Tokens per chunk (1 tile height)
constexpr uint32_t TOKENS_PER_CHUNK = 32;

// GPT-OSS dimensions
constexpr uint32_t HIDDEN_SIZE = 2880;
constexpr uint32_t INTERMEDIATE_SIZE = 2880;
constexpr uint32_t EXPERTS_PER_DEVICE = 4;

// Tile counts
constexpr uint32_t NUM_W0_W1_TILES_H = HIDDEN_SIZE / 32;     // 90
constexpr uint32_t NUM_W2_TILES_H = INTERMEDIATE_SIZE / 32;  // 90
constexpr uint32_t K_TILES = HIDDEN_SIZE / 32;               // 90

// Transaction sizing for DRAM reads (same as moe_gpt)
constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
constexpr uint32_t W0_W1_TILES_PER_TXN = 10;

constexpr uint32_t W2_TXNS_PER_BLOCK = 2;
constexpr uint32_t W2_TILES_PER_TXN = 10;

constexpr uint32_t W2_BLOCKS_PER_EXPERT = 36;

// Ring cores per combine column: 12 / 3 = 4
constexpr uint32_t RING_CORES_PER_COMBINE_COL = NUM_CORES / COMBINE_WIDTH_SHARD_DIM;

//-----------------------------------------------------------------------------
// Precomputed lookup tables (identical to moe_gpt)
//-----------------------------------------------------------------------------

// Boundary-optimized: cores {0,1,4,5,8,9} get 8 tiles
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP_A[NUM_CORES][NUM_CORES] = {
    {8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8},
    {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7},
    {7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7},
    {7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8},
    {8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8},
    {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7},
    {7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7},
    {7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8},
    {8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8},
    {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7},
    {7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7},
    {7, 7, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8},
};

// W2 tiles per core
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

constexpr uint32_t IN2_TILES_PER_STEP_A = *std::max_element(
    W0_W1_TILES_PER_CORE_PER_STEP_A[0], W0_W1_TILES_PER_CORE_PER_STEP_A[0] + NUM_CORES, [](uint32_t a, uint32_t b) {
        return a < b;
    });

constexpr uint32_t NUM_A2A_ITERS_A = *std::max_element(
                                         W2_TILES_PER_CORE_A,
                                         W2_TILES_PER_CORE_A + NUM_CORES,
                                         [](uint32_t a, uint32_t b) { return (a / 2) < (b / 2); }) /
                                     4;

// Max output tiles per expert per core (for legacy reference)
constexpr uint32_t MAX_OUTPUT_TILES_PER_EXPERT = NUM_A2A_ITERS_A * 4;  // 8

// Combine output constants
// Each height shard = one expert = TOKENS_PER_CHUNK tokens
// Shard shape: [TOKENS_PER_CHUNK, K / COMBINE_WIDTH_SHARD_DIM] = [32, 960]
constexpr uint32_t COMBINE_SHARD_WIDTH_TILES = K_TILES / COMBINE_WIDTH_SHARD_DIM;  // 90/3 = 30
constexpr uint32_t SOURCE_WIDTH_TILES = IN2_TILES_PER_STEP_A;                      // 8 (max tiles per core)

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

// Tilize constants
constexpr uint32_t TILES_PER_GATHER_CORE = K_TILES / NUM_GATHER_CORES;       // 90/3 = 30
constexpr uint32_t TILIZE_INPUT_PAGE_SIZE = TILES_PER_GATHER_CORE * 32 * 2;  // 30 * 32 * 2 = 1920 bytes per row

}  // namespace moe_gpt_fused_ring
