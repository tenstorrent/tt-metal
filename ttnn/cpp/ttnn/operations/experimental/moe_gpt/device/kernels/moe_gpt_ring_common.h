// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <algorithm>

//=============================================================================
// MoEGPT Ring All-to-All Configuration for GPT-OSS
//
// GPT-OSS dimensions: K=2880 (90 tiles), N=2880 (90 tiles), 4 experts/device
// TILES_PER_TXN=10 (90/10=9 exact, no padding needed)
//
// Distribution: even-indexed source cores get 8 tiles, odd get 7
// Sum per row/column = 6*8 + 6*7 = 90
//=============================================================================

namespace moe_gpt_ring {

constexpr uint32_t NUM_CORES = 12;

constexpr uint32_t NUM_W0_W1_TILES_H = 90;  // K=2880, 2880/32=90
constexpr uint32_t NUM_W2_TILES_H = 90;     // N=2880, 2880/32=90

constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
constexpr uint32_t W0_W1_TILES_PER_TXN = 10;  // 90/10=9 exact

constexpr uint32_t W2_TXNS_PER_BLOCK = 2;
constexpr uint32_t W2_TILES_PER_TXN = 10;  // 90/10=9 exact

constexpr uint32_t W2_BLOCKS_PER_EXPERT = 36;  // num_a2a_iters(2) * w2_blocks_per_four_mm2_tile(18)

// Static assertions: TILES_PER_TXN must divide TILES_H evenly
static_assert(NUM_W0_W1_TILES_H % W0_W1_TILES_PER_TXN == 0, W0_W1_TILES_H must be divisible by W0_W1_TILES_PER_TXN);
static_assert(NUM_W2_TILES_H % W2_TILES_PER_TXN == 0, W2_TILES_H must be divisible by W2_TILES_PER_TXN);

//-----------------------------------------------------------------------------
// Precomputed lookup tables
// Distribution: even-indexed source cores get 8 N-tiles, odd get 7
// table[core_id][step] = tiles_owned_by[(core_id - step) % 12]
//-----------------------------------------------------------------------------

// Pattern A: even source cores get more tiles
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP_A[NUM_CORES][NUM_CORES] = {
    // Core 0 (even): sources [0,11,10,9,8,7,6,5,4,3,2,1]
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    // Core 1 (odd): sources [1,0,11,10,9,8,7,6,5,4,3,2]
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    // Core 2 (even)
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    // Core 3 (odd)
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    // Core 4 (even)
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    // Core 5 (odd)
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    // Core 6 (even)
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    // Core 7 (odd)
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    // Core 8 (even)
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    // Core 9 (odd)
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    // Core 10 (even)
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    // Core 11 (odd)
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
};

// Pattern B: complement (odd source cores get more tiles)
constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP_B[NUM_CORES][NUM_CORES] = {
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
    {7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8},
    {8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7},
};

// W2 K-column tiles per core (same distribution as W0_W1 since K=N=2880)
constexpr uint32_t W2_TILES_PER_CORE_A[NUM_CORES] = {
    8,
    7,
    8,
    7,
    8,
    7,
    8,
    7,
    8,
    7,
    8,
    7,
};

constexpr uint32_t W2_TILES_PER_CORE_B[NUM_CORES] = {
    7,
    8,
    7,
    8,
    7,
    8,
    7,
    8,
    7,
    8,
    7,
    8,
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

}  // namespace moe_gpt_ring
