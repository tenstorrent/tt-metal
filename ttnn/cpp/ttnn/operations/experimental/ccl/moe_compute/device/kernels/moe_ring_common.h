// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <algorithm>
#include <stdint.h>

#include "../hostdevcommon/config.hpp"

namespace detail {
inline uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }

template <uint32_t a, uint32_t b>
constexpr uint32_t div_up() {
    return (a + b - 1) / b;
}

}  // namespace detail

namespace moe_ring {

constexpr uint32_t NUM_CORES = 12;  // Total number of cores in the ring

constexpr uint32_t W0_W1_TXNS_PER_BLOCK = 2;
constexpr uint32_t W0_W1_TILES_PER_TXN = 14;

// probably don't need this for W0_W1 and W2 because it's the same
constexpr uint32_t W0_W1_BLOCK_TILES_W = 4;
constexpr uint32_t W0_W1_BLOCK_TILES_H =
    (W0_W1_TXNS_PER_BLOCK * W0_W1_TILES_PER_TXN) / W0_W1_BLOCK_TILES_W;  // = (2 * 14) / 4 = 7

constexpr uint32_t W2_TXNS_PER_BLOCK = 2;
constexpr uint32_t W2_TILES_PER_TXN = 14;

constexpr uint32_t TOKENS_PER_CHUNK = 32;

// Let's call this a constant
constexpr uint32_t W2_TILES_PER_A2A_ITER_W = 4;
constexpr uint32_t W2_TILES_PER_A2A_ITER_H =
    (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN) / W2_TILES_PER_A2A_ITER_W;  // = (2 * 14) / 4 = 7

static constexpr uint32_t OUTPUT_HEIGHT_SHARD_DIM = 4;

//-----------------------------------------------------------------------------
// Precomputed lookup tables (generated at compile time)
// Use these if you prefer lookup over runtime computation
//-----------------------------------------------------------------------------

namespace detail {

constexpr uint32_t compute_a2a_iters(const uint32_t* arr) {
    return (*std::max_element(arr, arr + NUM_CORES) + W2_TILES_PER_A2A_ITER_W - 1) / W2_TILES_PER_A2A_ITER_W;
};

constexpr uint32_t compute_in2_tiles_per_step(const uint32_t* arr) { return *std::max_element(arr, arr + NUM_CORES); };

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

template <bool HasBias>
struct DeepSeekRingConfig {
    static constexpr uint32_t NUM_W0_W1_TILES_H = 224;     // Height of W0/W1 weight matrix in tiles (7168 / 32 = 224)
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;  // 225 or 224
    static constexpr uint32_t NUM_W2_TILES_H = 64;         // Height of W2 weight matrix in tiles (2048 / 32 = 64)
    static constexpr uint32_t NUM_W2_DRAM_TILES_H = (HasBias) ? NUM_W2_TILES_H + 1 : NUM_W2_TILES_H;  // 65 or 64
    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 4;  // Number of data-parallel shards for output

    // Evenly distributed: tiles[core_id][step]
    // Each row represents how many tiles a core receives at each step of the ring all-to-all
    // Total tiles per step across all cores = 64 (alternates between 4*6+8*5=64 and 8*6+4*5=68)
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

    // W2 tiles per core distribution for width dimension
    // Pattern: [18,19,19] repeated 4 times (4 cores get 18, 8 cores get 19)
    // These are width tiles per core for the W2 matrix
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

    static constexpr auto NUM_A2A_ITERS = detail::compute_a2a_iters(
        W2_TILES_PER_CORE);  // = max(18,19,19,18,19,19,18,19,19,18,19,19) / 4 = 19 / 4 = 5 (rounded up)

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = W2_TILES_PER_A2A_ITER_W * NUM_A2A_ITERS;  // = 4 * 5 = 20
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        W2_TILES_PER_A2A_ITER_H * ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) /
                                   W2_TILES_PER_A2A_ITER_H);  // = 7 * ((64 + 7 - 1) / 7) = 7 * 10 = 70

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (20 * 70) / (2 * 14) = 1400 / 28 = 50

    static constexpr auto IN2_TILES_PER_STEP =
        detail::compute_in2_tiles_per_step(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = max(6,5,5,6,5,5,6,5,5,6,5,5) = 6

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 18, 37, 56, 74, 93, 112, 130, 149, 168, 186, 205]
};

// For GPT-OSS: K = N = 2880, so both W0/W1 and W2 have 90x90 tile dimensions
template <bool HasBias>
struct GptRingConfig {
    static constexpr uint32_t NUM_W0_W1_TILES_H = 90;  // Height of W0/W1 weight matrix in tiles (2880 / 32 = 90)
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;  // 91 or 90

    static constexpr uint32_t NUM_W2_TILES_H = 90;     // Height of W2 weight matrix in tiles (2880 / 32 = 90)
    static constexpr uint32_t NUM_W2_DRAM_TILES_H = (HasBias) ? NUM_W2_TILES_H + 1 : NUM_W2_TILES_H;  // 91 or 90

    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 3;  // Number of data-parallel shards for output

    // Boundary-optimized: tiles[core_id][step] - cores {0,1,4,5,8,9} get 8 tiles when source%4 < 2
    // Each row represents how many tiles a core receives at each step of the ring all-to-all
    // Total tiles per step across all cores = 90 (6 cores * 8 tiles + 6 cores * 7 tiles = 48 + 42 = 90)
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
    // Pattern: [8,8,7,7] repeated 3 times
    // Total: 6*8 + 6*7 = 48 + 42 = 90 tiles
    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7};

    static constexpr auto IN2_TILES_PER_STEP =
        detail::compute_in2_tiles_per_step(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = max(8,7,7,8,...) = 8

    static constexpr auto NUM_A2A_ITERS =
        detail::compute_a2a_iters(W2_TILES_PER_CORE);  // = max(8,8,7,7,8,8,7,7,8,8,7,7) / 4 = 8 / 4 = 2

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = NUM_A2A_ITERS * W2_TILES_PER_A2A_ITER_W;  // = 2 * 4 = 8
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) / W2_TILES_PER_A2A_ITER_H) *
        W2_TILES_PER_A2A_ITER_H;  // = ((90 + 7 - 1) / 7) * 7 = 13 * 7 = 91

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (8 * 91) / (2 * 14) = 728 / 28 = 26

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 8, 16, 23, 30, 38, 46, 53, 60, 68, 76, 83]
};

// Template trait for config type selection
template <bool HasBias, ttnn::experimental::prim::detail::MoEConfigType ConfigType>
struct ConfigTypeSelector;

// Template specialization for DeepSeek
template <bool HasBias>
struct ConfigTypeSelector<HasBias, ttnn::experimental::prim::detail::MoEConfigType::DEEPSEEK> {
    using type = DeepSeekRingConfig<HasBias>;
};

// Template specialization for GPT
template <bool HasBias>
struct ConfigTypeSelector<HasBias, ttnn::experimental::prim::detail::MoEConfigType::GPT> {
    using type = GptRingConfig<HasBias>;
};

// Helper alias template
template <bool HasBias, ttnn::experimental::prim::detail::MoEConfigType ConfigType>
using ConfigType_t = typename ConfigTypeSelector<HasBias, ConfigType>::type;

}  // namespace moe_ring
