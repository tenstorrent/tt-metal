// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

constexpr uint32_t compute_w2_tile_offset(uint32_t core_id, uint32_t Ht, uint32_t Nt, uint32_t n_cores) {
    uint32_t offset = 0;
    for (uint32_t i = 0; i < core_id; ++i) {
        offset += w2_shard_tiles(Ht, i, Nt, n_cores);
    }
    return offset;
}

}  // namespace moe_ring
