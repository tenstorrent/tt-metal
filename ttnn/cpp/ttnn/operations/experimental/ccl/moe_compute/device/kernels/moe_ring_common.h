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

// Ring core count is templatized via DeepSeekRingConfig<HasBias, N> / GptRingConfig<HasBias, N>.
// The kernel reads N from the named CT arg "num_cores" and selects the appropriate
// specialization at compile time. Both Wormhole and Blackhole currently instantiate <12>
// (BH pads its 8-DRAM-bank assignment with 4 extra cores and uses INTERLEAVED weights
// to decouple ring core count from bank count). See issue #41827 PR1 (BH N=12).

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

template <uint32_t N>
constexpr uint32_t compute_a2a_iters(const uint32_t* arr) {
    return (*std::max_element(arr, arr + N) + W2_TILES_PER_A2A_ITER_W - 1) / W2_TILES_PER_A2A_ITER_W;
};

template <uint32_t N>
constexpr uint32_t compute_in2_tiles_per_step(const uint32_t* arr) {
    return *std::max_element(arr, arr + N);
};

template <uint32_t N>
constexpr std::array<uint32_t, N> compute_combine_w_offset_per_core(const uint32_t* arr) {
    std::array<uint32_t, N> out = {};
    uint32_t sum = 0;
    for (uint32_t i = 0; i < N; ++i) {
        out[i] = sum;
        sum += arr[i];
    }
    return out;
}

}  // namespace detail

// Forward declaration for the templatized DeepSeek ring config.
// Currently specialized for N=8 (BH HEIGHT_SHARDED, M1 pattern), N=12, N=16. The N=8 path
// matches BH's 8 DRAM bank count and uses HEIGHT_SHARDED weights (same code path as WH).
// N=12/16 use INTERLEAVED weights to decouple ring core count from bank count. See issue
// #41827.
template <bool HasBias, uint32_t N>
struct DeepSeekRingConfig;

// DeepSeek config (8 matmul cores) — BH HEIGHT_SHARDED perf-experiment specialization.
// 8 ring cores match the 8 BH DRAM banks 1:1, so weights can be HEIGHT_SHARDED (M1 pattern,
// dm0.cpp set_state + with_state fast path) instead of INTERLEAVED. 64 W0/W1 tiles per step
// / 8 cores = 8 each (perfectly balanced); 224 W2 width tiles / 8 cores = 28 each. 8 /
// OUTPUT_WIDTH_SHARD_DIM(=4) = 2 ring cores per data-parallel column.
template <bool HasBias>
struct DeepSeekRingConfig<HasBias, 8> {
    static constexpr uint32_t NUM_CORES = 8;
    static constexpr uint32_t NUM_W0_W1_TILES_H = 224;  // 7168 / 32 = 224
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;                                        // 225 or 224
    static constexpr uint32_t NUM_W2_TILES_H = 64;                                                    // 2048 / 32 = 64
    static constexpr uint32_t NUM_W2_DRAM_TILES_H = (HasBias) ? NUM_W2_TILES_H + 1 : NUM_W2_TILES_H;  // 65 or 64
    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 4;

    // Evenly distributed: tiles[core_id][step]; 64/8 = 8 tiles per cell, perfectly balanced.
    // Total per step across cores = 8 * 8 = 64.
    static constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[NUM_CORES][NUM_CORES] = {
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
        {8, 8, 8, 8, 8, 8, 8, 8},
    };

    // W2 tiles per core: 224 / 8 = 28 each, perfectly balanced. (Total = 8 * 28 = 224.)
    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {28, 28, 28, 28, 28, 28, 28, 28};

    static constexpr auto NUM_A2A_ITERS = detail::compute_a2a_iters<NUM_CORES>(W2_TILES_PER_CORE);  // = max(28) / 4 = 7

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = W2_TILES_PER_A2A_ITER_W * NUM_A2A_ITERS;  // = 4 * 7 = 28
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        W2_TILES_PER_A2A_ITER_H *
        ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) / W2_TILES_PER_A2A_ITER_H);  // = 7 * ((64 + 6) / 7) = 70

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT = W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
                                                     (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (28 * 70) / 28 = 70

    static constexpr auto IN2_TILES_PER_STEP =
        detail::compute_in2_tiles_per_step<NUM_CORES>(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = 8

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core<NUM_CORES>(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 28, 56, 84, 112, 140, 168, 196]
};

// DeepSeek config (12 matmul cores). All values are pure functions of N — no arch-specific
// logic. Byte-identical tile tables to the pre-templatize M1 baseline.
template <bool HasBias>
struct DeepSeekRingConfig<HasBias, 12> {
    static constexpr uint32_t NUM_CORES = 12;
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

    static constexpr auto NUM_A2A_ITERS = detail::compute_a2a_iters<NUM_CORES>(
        W2_TILES_PER_CORE);  // = max(18,19,19,18,19,19,18,19,19,18,19,19) / 4 = 19 / 4 = 5 (rounded up)

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = W2_TILES_PER_A2A_ITER_W * NUM_A2A_ITERS;  // = 4 * 5 = 20
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        W2_TILES_PER_A2A_ITER_H * ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) /
                                   W2_TILES_PER_A2A_ITER_H);  // = 7 * ((64 + 7 - 1) / 7) = 7 * 10 = 70

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (20 * 70) / (2 * 14) = 1400 / 28 = 50

    static constexpr auto IN2_TILES_PER_STEP = detail::compute_in2_tiles_per_step<NUM_CORES>(
        W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = max(6,5,5,6,5,5,6,5,5,6,5,5) = 6

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core<NUM_CORES>(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 18, 37, 56, 74, 93, 112, 130, 149, 168, 186, 205]
};

// DeepSeek config (16 matmul cores) — BH N=16 perf experiment specialization.
// At N=16 the 64-tile-per-step DeepSeek workload distributes perfectly: every core gets
// exactly 4 W0/W1 tiles per step (no jitter pattern needed). The 224 W2 width tiles split
// evenly into 14 tiles per core. 16 / OUTPUT_WIDTH_SHARD_DIM(=4) = 4 ring cores per
// data-parallel column (same column count as the N=12 specialization, 12/4=3 → here 16/4=4).
template <bool HasBias>
struct DeepSeekRingConfig<HasBias, 16> {
    static constexpr uint32_t NUM_CORES = 16;
    static constexpr uint32_t NUM_W0_W1_TILES_H = 224;  // Height of W0/W1 weight matrix in tiles (7168 / 32 = 224)
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;  // 225 or 224
    static constexpr uint32_t NUM_W2_TILES_H = 64;              // Height of W2 weight matrix in tiles (2048 / 32 = 64)
    static constexpr uint32_t NUM_W2_DRAM_TILES_H = (HasBias) ? NUM_W2_TILES_H + 1 : NUM_W2_TILES_H;  // 65 or 64
    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 4;  // Number of data-parallel shards for output

    // Evenly distributed: tiles[core_id][step]
    // Each row represents how many tiles a core receives at each step of the ring all-to-all.
    // 64 W0/W1 tiles per step / 16 cores = 4 tiles per core (perfectly balanced).
    static constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[NUM_CORES][NUM_CORES] = {
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
        {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
    };

    // W2 tiles per core distribution: 224 W2 width tiles / 16 cores = 14 tiles each.
    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
        14,
    };

    static constexpr auto NUM_A2A_ITERS =
        detail::compute_a2a_iters<NUM_CORES>(W2_TILES_PER_CORE);  // = max(14) / 4 = ceil(14/4) = 4

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = W2_TILES_PER_A2A_ITER_W * NUM_A2A_ITERS;  // = 4 * 4 = 16
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        W2_TILES_PER_A2A_ITER_H * ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) /
                                   W2_TILES_PER_A2A_ITER_H);  // = 7 * ((64 + 7 - 1) / 7) = 7 * 10 = 70

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (16 * 70) / (2 * 14) = 1120 / 28 = 40

    static constexpr auto IN2_TILES_PER_STEP =
        detail::compute_in2_tiles_per_step<NUM_CORES>(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = max(4) = 4

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core<NUM_CORES>(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 196,
                             // 210]
};

// Forward declaration for the templatized GPT-OSS ring config.
template <bool HasBias, uint32_t N>
struct GptRingConfig;

// GPT-OSS config (8 matmul cores) — BH HEIGHT_SHARDED perf-experiment specialization.
// 90 W0_W1 tiles per step / 8 cores = 11.25 → 2 cores get 12, 6 cores get 11 (each row sums
// to 90 = 2*12 + 6*11). Rotate the "12" position by +1 per row so load-per-core averaged
// over the 8-step cycle is balanced. 8 / OUTPUT_WIDTH_SHARD_DIM(=4) = 2 ring cores per
// data-parallel column.
template <bool HasBias>
struct GptRingConfig<HasBias, 8> {
    static constexpr uint32_t NUM_CORES = 8;
    static constexpr uint32_t NUM_W0_W1_TILES_H = 90;  // 2880 / 32 = 90
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;  // 91 or 90

    static constexpr uint32_t NUM_W2_TILES_H = 90;                                                    // 2880 / 32 = 90
    static constexpr uint32_t NUM_W2_DRAM_TILES_H = (HasBias) ? NUM_W2_TILES_H + 1 : NUM_W2_TILES_H;  // 91 or 90

    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 4;  // 8 / 4 = 2 ring cores per data-parallel column

    // tiles[core_id][step] - 90 tiles per step / 8 cores = 11.25
    // Pattern: each row sums to 90 = 2*12 + 6*11. 2 cores get 12, 6 cores get 11.
    // Rotate the "12" position by +4 each row so load-per-core averaged over steps is balanced
    // (each core gets a "12" twice across the 8-step cycle).
    static constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[NUM_CORES][NUM_CORES] = {
        // Core 0: 12s at positions {0, 4} -> 12+11+11+11+12+11+11+11 = 90
        {12, 11, 11, 11, 12, 11, 11, 11},
        // Core 1: 12s at positions {1, 5}
        {11, 12, 11, 11, 11, 12, 11, 11},
        // Core 2: 12s at positions {2, 6}
        {11, 11, 12, 11, 11, 11, 12, 11},
        // Core 3: 12s at positions {3, 7}
        {11, 11, 11, 12, 11, 11, 11, 12},
        // Core 4: 12s at positions {0, 4} (cycle repeats every 4 rows)
        {12, 11, 11, 11, 12, 11, 11, 11},
        // Core 5: 12s at positions {1, 5}
        {11, 12, 11, 11, 11, 12, 11, 11},
        // Core 6: 12s at positions {2, 6}
        {11, 11, 12, 11, 11, 11, 12, 11},
        // Core 7: 12s at positions {3, 7}
        {11, 11, 11, 12, 11, 11, 11, 12},
    };

    // W2 tiles per core: 90 / 8 = 11.25 -> 2 cores get 12, 6 cores get 11. Total: 2*12 + 6*11 = 90.
    // Pattern: [12, 11, 11, 11, 12, 11, 11, 11] mirrors the W0_W1 row-0 pattern for consistency.
    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {12, 11, 11, 11, 12, 11, 11, 11};

    static constexpr auto IN2_TILES_PER_STEP =
        detail::compute_in2_tiles_per_step<NUM_CORES>(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = 12

    static constexpr auto NUM_A2A_ITERS =
        detail::compute_a2a_iters<NUM_CORES>(W2_TILES_PER_CORE);  // = max(12) / 4 = 3 (rounded up)

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = NUM_A2A_ITERS * W2_TILES_PER_A2A_ITER_W;  // = 3 * 4 = 12
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) / W2_TILES_PER_A2A_ITER_H) *
        W2_TILES_PER_A2A_ITER_H;  // = ((90 + 7 - 1) / 7) * 7 = 13 * 7 = 91

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (12 * 91) / (2 * 14) = 1092 / 28 = 39

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core<NUM_CORES>(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 12, 23, 34, 45, 57, 68, 79]
};

// GPT-OSS config (12 matmul cores). All values are pure functions of N. K = N = 2880 →
// both W0/W1 and W2 are 90x90 tiles. Byte-identical tile tables to the pre-templatize
// M1 baseline.
template <bool HasBias>
struct GptRingConfig<HasBias, 12> {
    static constexpr uint32_t NUM_CORES = 12;
    static constexpr uint32_t NUM_W0_W1_TILES_H = 90;  // 2880 / 32 = 90
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;  // 91 or 90

    static constexpr uint32_t NUM_W2_TILES_H = 90;                                                    // 2880 / 32 = 90
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
        detail::compute_in2_tiles_per_step<NUM_CORES>(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = max(8,7,7,8,...) = 8

    static constexpr auto NUM_A2A_ITERS =
        detail::compute_a2a_iters<NUM_CORES>(W2_TILES_PER_CORE);  // = max(8,8,7,7,8,8,7,7,8,8,7,7) / 4 = 8 / 4 = 2

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = NUM_A2A_ITERS * W2_TILES_PER_A2A_ITER_W;  // = 2 * 4 = 8
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) / W2_TILES_PER_A2A_ITER_H) *
        W2_TILES_PER_A2A_ITER_H;  // = ((90 + 7 - 1) / 7) * 7 = 13 * 7 = 91

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (8 * 91) / (2 * 14) = 728 / 28 = 26

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core<NUM_CORES>(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 8, 16, 23, 30, 38, 46, 53, 60, 68, 76, 83]
};

// GPT-OSS config (16 matmul cores) — BH N=16 perf experiment specialization.
// 90 W0_W1 tiles per step / 16 cores = 5.625 → 10 cores get 6, 6 cores get 5
// (10*6 + 6*5 = 90). The W2 width has the same split.
// NOTE: this distribution is balanced but NOT perf-tuned for ring stride. The DeepSeek
// path is the primary BH N=16 perf target; GPT N=16 is provided for completeness so
// the template machinery still resolves at hidden_size=2880. See issue #41827 N=16
// experiment.
template <bool HasBias>
struct GptRingConfig<HasBias, 16> {
    static constexpr uint32_t NUM_CORES = 16;
    static constexpr uint32_t NUM_W0_W1_TILES_H = 90;  // 2880 / 32 = 90
    static constexpr uint32_t NUM_W0_W1_DRAM_TILES_H =
        (HasBias) ? NUM_W0_W1_TILES_H + 1 : NUM_W0_W1_TILES_H;  // 91 or 90

    static constexpr uint32_t NUM_W2_TILES_H = 90;                                                    // 2880 / 32 = 90
    static constexpr uint32_t NUM_W2_DRAM_TILES_H = (HasBias) ? NUM_W2_TILES_H + 1 : NUM_W2_TILES_H;  // 91 or 90

    static constexpr uint32_t OUTPUT_WIDTH_SHARD_DIM = 4;  // 16 / 4 = 4 ring cores per data-parallel column

    // Stride-1 rotated pattern: row r has "6"s at positions (r+0..r+9) mod 16, "5"s at the rest.
    // Each step: 10*6 + 6*5 = 90 tiles total. Each core sees 10 "6"-steps and 6 "5"-steps over a
    // full ring rotation. Pattern is balanced but not stride-optimized for NoC traffic.
    static constexpr uint32_t W0_W1_TILES_PER_CORE_PER_STEP[NUM_CORES][NUM_CORES] = {
        // Core 0: "6"s at positions 0..9
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5},
        // Core 1: "6"s at positions 1..10
        {5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5},
        // Core 2: "6"s at positions 2..11
        {5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5},
        // Core 3: "6"s at positions 3..12
        {5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5},
        // Core 4: "6"s at positions 4..13
        {5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5},
        // Core 5: "6"s at positions 5..14
        {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5},
        // Core 6: "6"s at positions 6..15
        {5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6},
        // Core 7: "6"s at positions 7..15, 0
        {6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6},
        // Core 8: "6"s at positions 8..15, 0..1
        {6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6},
        // Core 9: "6"s at positions 9..15, 0..2
        {6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6},
        // Core 10: "6"s at positions 10..15, 0..3
        {6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6},
        // Core 11: "6"s at positions 11..15, 0..4
        {6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6},
        // Core 12: "6"s at positions 12..15, 0..5
        {6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6},
        // Core 13: "6"s at positions 13..15, 0..6
        {6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6, 6},
        // Core 14: "6"s at positions 14..15, 0..7
        {6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6, 6},
        // Core 15: "6"s at positions 15, 0..8
        {6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 6},
    };

    // W2 tiles per core: 90 / 16 = 5.625 → 10 cores get 6, 6 cores get 5 (10*6 + 6*5 = 90).
    // Cores {0..9} get 6, {10..15} get 5.
    static constexpr uint32_t W2_TILES_PER_CORE[NUM_CORES] = {
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        5,
        5,
        5,
        5,
        5,
        5,
    };

    static constexpr auto IN2_TILES_PER_STEP =
        detail::compute_in2_tiles_per_step<NUM_CORES>(W0_W1_TILES_PER_CORE_PER_STEP[0]);  // = max(6,5) = 6

    static constexpr auto NUM_A2A_ITERS =
        detail::compute_a2a_iters<NUM_CORES>(W2_TILES_PER_CORE);  // = max(6,5) / 4 = ceil(6/4) = 2

    static constexpr uint32_t W2_TILES_PER_EXPERT_W = NUM_A2A_ITERS * W2_TILES_PER_A2A_ITER_W;  // = 2 * 4 = 8
    static constexpr uint32_t W2_TILES_PER_EXPERT_H =
        ((NUM_W2_DRAM_TILES_H + W2_TILES_PER_A2A_ITER_H - 1) / W2_TILES_PER_A2A_ITER_H) *
        W2_TILES_PER_A2A_ITER_H;  // = ((90 + 7 - 1) / 7) * 7 = 13 * 7 = 91

    static constexpr uint32_t W2_BLOCKS_PER_EXPERT =
        W2_TILES_PER_EXPERT_W * W2_TILES_PER_EXPERT_H /
        (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);  // = (8 * 91) / (2 * 14) = 728 / 28 = 26

    static constexpr auto COMBINE_W_OFFSET_PER_CORE = detail::compute_combine_w_offset_per_core<NUM_CORES>(
        W2_TILES_PER_CORE);  // Cumulative offsets: [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 65, 70, 75, 80, 85]
};

// Template trait for config type selection. ConfigType picks DeepSeek vs GPT, N picks ring size.
template <bool HasBias, ttnn::experimental::prim::detail::MoEConfigType ConfigType, uint32_t N>
struct ConfigTypeSelector;

// Template specialization for DeepSeek
template <bool HasBias, uint32_t N>
struct ConfigTypeSelector<HasBias, ttnn::experimental::prim::detail::MoEConfigType::DEEPSEEK, N> {
    using type = DeepSeekRingConfig<HasBias, N>;
};

// Template specialization for GPT
template <bool HasBias, uint32_t N>
struct ConfigTypeSelector<HasBias, ttnn::experimental::prim::detail::MoEConfigType::GPT, N> {
    using type = GptRingConfig<HasBias, N>;
};

// Helper alias template
template <bool HasBias, ttnn::experimental::prim::detail::MoEConfigType ConfigType, uint32_t N>
using ConfigType_t = typename ConfigTypeSelector<HasBias, ConfigType, N>::type;

}  // namespace moe_ring
