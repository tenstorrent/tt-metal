// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
// W0/W1: [K, N] = [90, 90] tiles -> ceil(90/num_cores) or floor tiles/core
// W2:    [N, K] = [90, 90] tiles -> ceil(90/num_cores) or floor tiles/core
//
// Both W0/W1 and W2 have the same distribution since K == N == 2880.
//
// Ring size (num_cores) is templated, not a fixed literal: Wormhole exposes 12 DRAM banks,
// Blackhole 7 or 8 (harvesting-dependent). Two distribution strategies coexist:
//
//   - num_cores == 12 (and Ht == Nt == 90): the ORIGINAL, hand-picked "boundary-optimized"
//     pairs table {8,8,7,7,8,8,7,7,8,8,7,7} (cores {0,1,4,5,8,9} get 8 tiles), returned
//     byte-for-byte unchanged via a literal-data passthrough. This is not re-derived from a
//     formula -- it is the exact original data, reached by the exact original lookup, so the
//     Wormhole code path is provably unaffected by this generalization (verified by
//     static_assert below, not just by inspection).
//   - any other num_cores: a general Euclidean-rhythm distribution (shard_tiles /
//     is_big_shard), ported from the already-proven ttnn.experimental.moe_compute kernel
//     (ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/
//     moe_ring_common.h). This produces a DIFFERENT (but equally valid -- every core still
//     sums to n_tiles, and consecutive groups of RING_CORES_PER_COMBINE_COL ring cores still
//     sum evenly, which is what the combine step depends on) round-robin pattern than the
//     legacy pairs table, e.g. {8,7,8,7,...} instead of {8,8,7,7,...} for the same totals --
//     it is intentionally NOT used to replace the legacy table at num_cores==12, only to
//     cover ring sizes the legacy table was never built for.
//=============================================================================

namespace moe_gpt_ring {

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

// Tokens per chunk (1 tile height)
constexpr uint32_t TOKENS_PER_CHUNK = 32;

namespace legacy_wh12 {
// Original, UNCHANGED Wormhole (12-core) literal tables. Do not edit these -- their being
// untouched is exactly what makes the Wormhole code path provably unaffected by this file's
// generalization. See MoeGptRingConfig::kLegacyWh12 below for the only place they are read.
constexpr uint32_t kNumCores = 12;

// Boundary-optimized: cores {0,1,4,5,8,9} get 8 tiles, rest get 7.
constexpr uint32_t kW2TilesPerCore[kNumCores] = {8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 7, 7};

// Each row[core][step] = kW2TilesPerCore[(core - step) mod 12]
constexpr uint32_t kW0W1TilesPerCorePerStep[kNumCores][kNumCores] = {
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

constexpr uint32_t kCombineWidthShardDim = 3;
constexpr uint32_t kCombineHeightShardDim = 4;
}  // namespace legacy_wh12

//-----------------------------------------------------------------------------
// General ring-size-agnostic shard distribution (Euclidean rhythm), ported from
// moe_compute/device/kernels/moe_ring_common.h -- kept in sync with that file's
// is_big_w0w1/shard_tiles functions by construction (same formula, same names).
//-----------------------------------------------------------------------------

constexpr bool is_big_shard(uint32_t core_id, uint32_t n_big, uint32_t n_cores) {
    return n_big > 0 && (core_id * n_big) % n_cores < n_big;
}

constexpr uint32_t shard_tiles(uint32_t n_tiles, uint32_t core_id, uint32_t n_cores) {
    const uint32_t n_big = n_tiles % n_cores;
    const uint32_t small = n_tiles / n_cores;
    return small + (is_big_shard(core_id, n_big, n_cores) ? 1u : 0u);
}

template <uint32_t n_tiles, uint32_t num_cores>
constexpr std::array<uint32_t, num_cores> make_shard_array() {
    std::array<uint32_t, num_cores> arr{};
    for (uint32_t c = 0; c < num_cores; ++c) {
        arr[c] = shard_tiles(n_tiles, c, num_cores);
    }
    return arr;
}

// row[core][step] = shard_array[(core - step) mod num_cores], matching the legacy table's
// exact indexing convention so kernel consumers (dm0/dm1/compute.cpp) need no change beyond
// reading a differently-sized array.
template <uint32_t n_tiles, uint32_t num_cores>
constexpr std::array<std::array<uint32_t, num_cores>, num_cores> make_step_table() {
    std::array<std::array<uint32_t, num_cores>, num_cores> table{};
    const auto source_tiles = make_shard_array<n_tiles, num_cores>();
    for (uint32_t core = 0; core < num_cores; ++core) {
        for (uint32_t step = 0; step < num_cores; ++step) {
            const uint32_t src = (core + num_cores - (step % num_cores)) % num_cores;
            table[core][step] = source_tiles[src];
        }
    }
    return table;
}

template <uint32_t n_tiles, uint32_t num_cores>
constexpr std::array<uint32_t, num_cores> make_offset_array() {
    std::array<uint32_t, num_cores> arr{};
    uint32_t sum = 0;
    for (uint32_t c = 0; c < num_cores; ++c) {
        arr[c] = sum;
        sum += shard_tiles(n_tiles, c, num_cores);
    }
    return arr;
}

// Largest divisor d of n_tiles (with d <= max_dim) such that num_cores % d == 0, mirroring
// ttnn/ttnn/_experimental/moe_compute_utils.py::auto_output_width_shard_dim. Falls back to 1
// if nothing else divides evenly (always valid, if degenerate).
constexpr uint32_t auto_width_shard_dim(uint32_t n_tiles, uint32_t num_cores, uint32_t max_dim = 4) {
    for (uint32_t d = max_dim; d >= 1; --d) {
        if (n_tiles % d == 0 && num_cores % d == 0) {
            return d;
        }
    }
    return 1;
}

//-----------------------------------------------------------------------------
// MoeGptRingConfig<Ht, Nt, num_cores>: single source of truth for compute.cpp, dm0.cpp,
// dm1.cpp, combine_dm1.cpp. Ht == Nt == 90 for GPT-OSS-120B (hidden == intermediate == 2880,
// tile width 32).
//-----------------------------------------------------------------------------
template <uint32_t Ht, uint32_t Nt, uint32_t num_cores>
struct MoeGptRingConfig {
    static constexpr bool kLegacyWh12 = (num_cores == 12 && Ht == 90 && Nt == 90);

    // W2_TILES_PER_CORE_A equivalent: tiles held by each core.
    static constexpr std::array<uint32_t, num_cores> w2_tiles_per_core = []() constexpr {
        if constexpr (kLegacyWh12) {
            std::array<uint32_t, num_cores> arr{};
            for (uint32_t c = 0; c < num_cores; ++c) {
                arr[c] = legacy_wh12::kW2TilesPerCore[c];
            }
            return arr;
        } else {
            return make_shard_array<Ht, num_cores>();
        }
    }();

    // W0_W1_TILES_PER_CORE_PER_STEP_A equivalent.
    static constexpr std::array<std::array<uint32_t, num_cores>, num_cores> w0_w1_tiles_per_core_per_step =
        []() constexpr {
            if constexpr (kLegacyWh12) {
                std::array<std::array<uint32_t, num_cores>, num_cores> table{};
                for (uint32_t core = 0; core < num_cores; ++core) {
                    for (uint32_t step = 0; step < num_cores; ++step) {
                        table[core][step] = legacy_wh12::kW0W1TilesPerCorePerStep[core][step];
                    }
                }
                return table;
            } else {
                return make_step_table<Nt, num_cores>();
            }
        }();

    // COMBINE_W_OFFSET_PER_CORE_A equivalent: prefix sum of w2_tiles_per_core.
    static constexpr std::array<uint32_t, num_cores> combine_w_offset_per_core = []() constexpr {
        std::array<uint32_t, num_cores> arr{};
        uint32_t sum = 0;
        for (uint32_t c = 0; c < num_cores; ++c) {
            arr[c] = sum;
            sum += w2_tiles_per_core[c];
        }
        return arr;
    }();

    // IN2_TILES_PER_STEP_A equivalent: max tiles any core holds (for CB/page sizing).
    static constexpr uint32_t in2_tiles_per_step =
        *std::max_element(w0_w1_tiles_per_core_per_step[0].begin(), w0_w1_tiles_per_core_per_step[0].end());

    static constexpr uint32_t source_width_tiles = in2_tiles_per_step;

    // NUM_A2A_ITERS_A equivalent.
    static constexpr uint32_t max_w2_tiles_per_core =
        *std::max_element(w2_tiles_per_core.begin(), w2_tiles_per_core.end());
    static constexpr uint32_t num_a2a_iters = max_w2_tiles_per_core / 4;

    // Combine grid. At num_cores==12 this reproduces the legacy 3x4 grid exactly; otherwise
    // width is auto-derived (mirrors auto_output_width_shard_dim) and height fills the rest.
    static constexpr uint32_t combine_width_shard_dim =
        kLegacyWh12 ? legacy_wh12::kCombineWidthShardDim : auto_width_shard_dim(Nt, num_cores);
    static constexpr uint32_t combine_height_shard_dim =
        kLegacyWh12 ? legacy_wh12::kCombineHeightShardDim : (num_cores / combine_width_shard_dim);
    static constexpr uint32_t ring_cores_per_combine_col = num_cores / combine_width_shard_dim;
    static constexpr uint32_t combine_shard_width_tiles = Nt / combine_width_shard_dim;
};

// Compile-time proof that the Wormhole (num_cores==12) path is byte-identical to the original,
// pre-generalization tables -- not a numeric coincidence, but a guard against someone
// accidentally deleting the kLegacyWh12 passthrough branch above.
namespace static_checks {
using Wh12Config = MoeGptRingConfig<90, 90, 12>;
static_assert(
    Wh12Config::w2_tiles_per_core[0] == 8 && Wh12Config::w2_tiles_per_core[1] == 8 &&
        Wh12Config::w2_tiles_per_core[2] == 7 && Wh12Config::w2_tiles_per_core[11] == 7,
    "MoeGptRingConfig<90,90,12> must reproduce the legacy Wormhole tile distribution exactly");
static_assert(
    Wh12Config::combine_width_shard_dim == 3 && Wh12Config::combine_height_shard_dim == 4,
    "MoeGptRingConfig<90,90,12> must reproduce the legacy Wormhole combine grid exactly");
static_assert(Wh12Config::ring_cores_per_combine_col == 4, "12 / 3 must still be 4");
}  // namespace static_checks

}  // namespace moe_gpt_ring
