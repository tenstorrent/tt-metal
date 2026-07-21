// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "../hostdevcommon/config.hpp"

namespace moe_ring {

namespace detail {
inline uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }

template <uint32_t a, uint32_t b>
constexpr uint32_t div_up() {
    return (a + b - 1) / b;
}

}  // namespace detail

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

//-----------------------------------------------------------------------------
// Shard distribution functions (hardware-agnostic).
// Identical to the Python equivalents in ttnn/ttnn/_experimental/moe_compute_utils.py.
//-----------------------------------------------------------------------------

constexpr bool is_big_w0w1(uint32_t core_id, uint32_t n_big, uint32_t n_cores) {
    return n_big > 0 && (core_id * n_big) % n_cores < n_big;
}

constexpr uint32_t shard_tiles(uint32_t n_tiles, uint32_t core_id, uint32_t n_cores) {
    const uint32_t n_big = n_tiles % n_cores;
    const uint32_t small = n_tiles / n_cores;
    return small + (is_big_w0w1(core_id, n_big, n_cores) ? 1u : 0u);
}

constexpr uint32_t w2_shard_tiles(uint32_t Ht, uint32_t core_id, uint32_t Nt, uint32_t n_cores) {
    const uint32_t n_big_nt = Nt % n_cores;
    const uint32_t n_big_ht = Ht % n_cores;
    const uint32_t small_ht = Ht / n_cores;
    if (n_big_nt + n_big_ht == n_cores) {
        return is_big_w0w1(core_id, n_big_nt, n_cores) ? small_ht : small_ht + 1u;
    }
    return shard_tiles(Ht, core_id, n_cores);
}

template <uint32_t N>
struct ShardLUT {
    uint32_t data[N];
    constexpr uint32_t operator[](uint32_t i) const { return data[i]; }
};

template <uint32_t n_tiles, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_shard_lut() {
    ShardLUT<n_cores> lut{};
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = shard_tiles(n_tiles, c, n_cores);
    }
    return lut;
}

template <uint32_t Ht, uint32_t Nt, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_w2_shard_lut() {
    ShardLUT<n_cores> lut{};
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = w2_shard_tiles(Ht, c, Nt, n_cores);
    }
    return lut;
}

template <uint32_t Ht, uint32_t Nt, uint32_t n_cores>
constexpr ShardLUT<n_cores> make_w2_offset_lut() {
    ShardLUT<n_cores> lut{};
    uint32_t offset = 0;
    for (uint32_t c = 0; c < n_cores; ++c) {
        lut.data[c] = offset;
        offset += w2_shard_tiles(Ht, c, Nt, n_cores);
    }
    return lut;
}

template <uint32_t Nt, bool has_bias, uint32_t W2TilesPerExpertW, uint32_t SharedExpertTp = 1>
constexpr uint32_t get_w2_blocks_per_expert() {
    constexpr uint32_t TpNt = moe_ring::detail::div_up<Nt, SharedExpertTp>();
    constexpr uint32_t w2_dram_tiles_h = has_bias ? TpNt + 1 : TpNt;
    constexpr uint32_t w2_tiles_per_expert_h =
        ((w2_dram_tiles_h + W2_TILES_PER_A2A_ITER_H - 1) / W2_TILES_PER_A2A_ITER_H) * W2_TILES_PER_A2A_ITER_H;

    return W2TilesPerExpertW * w2_tiles_per_expert_h / (W2_TXNS_PER_BLOCK * W2_TILES_PER_TXN);
}

//-----------------------------------------------------------------------------
// Derived ring constants — single source of truth for compute, dm0, dm1.
//-----------------------------------------------------------------------------
template <uint32_t Ht, uint32_t Nt, uint32_t num_cores, bool has_bias, uint32_t SharedExpertTp = 1>
struct MoeRingConfig {
    // W0/W1
    static constexpr uint32_t w0_w1_dram_tiles_h = has_bias ? Ht + 1 : Ht;
    static constexpr uint32_t w0_w1_blocks_per_col =
        (w0_w1_dram_tiles_h + W0_W1_BLOCK_TILES_H - 1) / W0_W1_BLOCK_TILES_H;
    static constexpr uint32_t in2_tiles_per_step = (((Nt + num_cores - 1) / num_cores) + 1) & ~1u;
    static constexpr uint32_t w0_w1_blocks_per_expert = w0_w1_blocks_per_col * in2_tiles_per_step / 2;

    // Shared-expert (TpNt) variants: the intermediate dim is TP-split to TpNt = ceil(Nt/tp).
    // After add_shared_expert_weights front-packs each core's real TpNt slice to the front of its
    // full-Nt shard, the kernel reads/produces only the real prefix (in2_tiles_per_step_shared per
    // core) and zero-fills the remainder of the full stride; the full W2 walk then contracts
    // real×real in the prefix and zero×zero past it.
    static constexpr uint32_t TpNt = detail::div_up<Nt, SharedExpertTp>();
    static constexpr uint32_t in2_tiles_per_step_shared = (((TpNt + num_cores - 1) / num_cores) + 1) & ~1u;
    static constexpr uint32_t w0_w1_blocks_per_shared_expert = w0_w1_blocks_per_col * in2_tiles_per_step_shared / 2;

    // W2
    static constexpr uint32_t max_w2_tiles_per_core = (Ht + num_cores - 1) / num_cores;
    static constexpr uint32_t num_a2a_iters =
        (max_w2_tiles_per_core + W2_TILES_PER_A2A_ITER_W - 1) / W2_TILES_PER_A2A_ITER_W;
    static constexpr uint32_t w2_tiles_per_expert_w = num_a2a_iters * W2_TILES_PER_A2A_ITER_W;
    static constexpr uint32_t w2_blocks_per_expert = get_w2_blocks_per_expert<Nt, has_bias, w2_tiles_per_expert_w>();
};

}  // namespace moe_ring
