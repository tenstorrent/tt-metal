// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @file threefry_key.hpp
 * @brief Threefry-2x32, used as the key-derivation PRF for rand.
 *
 * Threefry is the counter-based generator from Salmon et al., "Parallel Random Numbers: As Easy as
 * 1, 2, 3" (SC11), as published in Random123; the rotation table and key parity below are its
 * constants and the 20-round default is `THREEFRY2x32_DEFAULT_ROUNDS`. Deriving stream keys by
 * evaluating a keyed PRF at an index is how Random123 and JAX's `random.split` produce independent
 * substreams, and it is the same construction the device generator uses, so rand has exactly one
 * algorithm rather than a hand-rolled key schedule.
 *
 * Host-safe on purpose: the per-core keys the program factory derives and the epoch/position folding
 * the kernel does must agree bit-for-bit, so both sides call this one definition.
 */

namespace compute_kernel_lib {

struct ThreefryWords {
    std::uint32_t x0;
    std::uint32_t x1;
};

constexpr std::uint32_t threefry_key_parity = 0x1BD11BDAu;
constexpr int threefry_rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};

template <int ROUNDS = 20>
constexpr ThreefryWords threefry2x32(std::uint32_t ctr0, std::uint32_t ctr1, std::uint32_t key0, std::uint32_t key1) {
    const std::uint32_t ks[3] = {key0, key1, key0 ^ key1 ^ threefry_key_parity};
    std::uint32_t x0 = ctr0 + ks[0];
    std::uint32_t x1 = ctr1 + ks[1];
    for (int r = 0; r < ROUNDS; ++r) {
        x0 += x1;
        const int rot = threefry_rotations[r & 7];
        x1 = (x1 << rot) | (x1 >> (32 - rot));
        x1 ^= x0;
        if ((r & 3) == 3) {
            const int j = (r >> 2) + 1;
            x0 += ks[j % 3];
            x1 += ks[(j + 1) % 3] + static_cast<std::uint32_t>(j);
        }
    }
    return {x0, x1};
}

// The three places rand needs a derived key. Each is one PRF evaluation at a distinct counter, so the
// streams are disjoint by construction rather than by choice of mixing constants.
constexpr std::uint32_t rand_core_key(std::uint32_t seed, std::uint32_t shard_index, std::uint32_t core_index) {
    return threefry2x32(core_index, 0u, seed, shard_index).x0;
}
// Threefry's own key must not carry a core index -- that is what makes its output independent of the
// work split -- so it takes a counter slot no per-core key can reach.
constexpr std::uint32_t rand_stream_key(std::uint32_t seed, std::uint32_t shard_index) {
    return threefry2x32(0u, 1u, seed, shard_index).x0;
}
constexpr ThreefryWords rand_epoch_key(std::uint32_t core_key, std::uint32_t epoch_lo, std::uint32_t epoch_hi) {
    return threefry2x32(epoch_lo, epoch_hi, core_key, 0u);
}
// 13 rounds, not the 20 used for generation: this value only decorrelates equal LFSR states across
// tile positions, and 13 is Random123's crush-resistant round count. 20 rounds here costs 10.5% of the
// LFSR path's kernel time (4096^2 fp32 to DRAM, p150) against 1.2% for 13.
constexpr std::uint32_t rand_tile_salt(std::uint32_t core_key, std::uint32_t tile_index) {
    return threefry2x32<13>(tile_index, 0u, core_key, 1u).x0;
}

}  // namespace compute_kernel_lib
