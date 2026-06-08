// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

// Regression + quality tests for the ttsl::hash combiner (tt_stl/reflection.hpp).
//
// Background: the original combiner was the 32-bit boost::hash_combine, whose weak avalanche
// let distinct integer sequences collide in 64 bits. That manifested as wrong program-cache
// hits in issue #45821 (PermuteDeviceOperation reused a cached program built for a different
// shape). The canonical reproducer is the pair of tensor shapes below.
namespace ttsl::hash {
namespace {

// Hash a shape the same way the device-op hash path does: as a span of dimensions.
hash_t hash_shape(const std::vector<uint32_t>& dims) {
    return hash_objects_with_default_seed(ttsl::Span<const uint32_t>(dims.data(), dims.size()));
}

// --- Regression: the exact collision from issue #45821 -------------------------------------

TEST(HashCollisionTest, Issue45821_ShapesDoNotCollide) {
    // These two shapes flow through the same ttnn::permute path inside batch_norm; under the
    // old combiner they produced an identical 64-bit hash, so the second op wrongly hit the
    // first op's cached program.
    EXPECT_NE(hash_shape({3, 17, 1, 1}), hash_shape({1, 152, 1, 1}));
}

TEST(HashCollisionTest, Issue45821_TruncatedPrefixDoesNotCollide) {
    // The collision is already present at the 2-element prefix [3,17] vs [1,152] -- the trailing
    // [1, 1] is incidental. Pin the prefix too so a future regression is localized.
    EXPECT_NE(hash_shape({3, 17}), hash_shape({1, 152}));
}

// --- Order sensitivity ---------------------------------------------------------------------

TEST(HashCollisionTest, OrderMatters) {
    // A hash used as a cache key must distinguish permutations of the same multiset of dims;
    // otherwise e.g. a [2, 3] tensor and a [3, 2] tensor would share a program.
    EXPECT_NE(hash_shape({2, 3, 5}), hash_shape({5, 3, 2}));
    EXPECT_NE(hash_shape({1, 2}), hash_shape({2, 1}));
}

// --- Determinism ---------------------------------------------------------------------------

TEST(HashCollisionTest, Deterministic) {
    // The same input must hash identically across calls (cache hits depend on it).
    EXPECT_EQ(hash_shape({3, 17, 1, 1}), hash_shape({3, 17, 1, 1}));
    EXPECT_EQ(
        hash_objects_with_default_seed(uint32_t{42}, uint32_t{7}),
        hash_objects_with_default_seed(uint32_t{42}, uint32_t{7}));
}

// --- Quality sweep: no collisions over realistic small 4-D shapes --------------------------

TEST(HashCollisionTest, NoCollisionsOverSmall4DShapes) {
    // Sweep all 4-D shapes with each dim in [0, 40). The old combiner collided ~94k times over
    // this set; the splitmix64-based combiner must be collision-free here.
    constexpr uint32_t kMax = 40;
    std::unordered_set<hash_t> seen;
    seen.reserve(kMax * kMax * kMax * kMax);

    size_t count = 0;
    for (uint32_t a = 0; a < kMax; ++a) {
        for (uint32_t b = 0; b < kMax; ++b) {
            for (uint32_t c = 0; c < kMax; ++c) {
                for (uint32_t d = 0; d < kMax; ++d) {
                    seen.insert(hash_shape({a, b, c, d}));
                    ++count;
                }
            }
        }
    }
    EXPECT_EQ(seen.size(), count) << "hash collision detected among small 4-D shapes";
}

// --- Quality sweep: scalar pairs (the smallest structured keys) ----------------------------

TEST(HashCollisionTest, NoCollisionsOverScalarPairs) {
    constexpr uint32_t kMax = 256;
    std::unordered_set<hash_t> seen;
    seen.reserve(kMax * kMax);

    size_t count = 0;
    for (uint32_t a = 0; a < kMax; ++a) {
        for (uint32_t b = 0; b < kMax; ++b) {
            seen.insert(hash_objects_with_default_seed(a, b));
            ++count;
        }
    }
    EXPECT_EQ(seen.size(), count) << "hash collision detected among scalar pairs";
}

// Adversarial values: powers of two and values that differ only in high bits are exactly the
// inputs that defeat shift-and-add combiners (the high bits never propagate down).
TEST(HashCollisionTest, NoCollisionsOverHighBitAndPowerOfTwoShapes) {
    std::unordered_set<uint32_t> unique_vals;
    for (uint32_t shift = 0; shift < 32; ++shift) {
        unique_vals.insert(1u << shift);          // powers of two
        unique_vals.insert((1u << shift) - 1u);   // all-low-bits-set
    }
    unique_vals.insert(0xFFFFFFFFu);
    // Dedupe first: (1<<1)-1 == 1<<0 etc. would otherwise produce identical shapes that are not
    // collisions. We want collisions among DISTINCT shapes only.
    const std::vector<uint32_t> vals(unique_vals.begin(), unique_vals.end());

    std::unordered_set<hash_t> seen;
    size_t count = 0;
    for (uint32_t a : vals) {
        for (uint32_t b : vals) {
            for (uint32_t c : vals) {
                seen.insert(hash_shape({a, b, c}));
                ++count;
            }
        }
    }
    EXPECT_EQ(seen.size(), count) << "collision among power-of-two / high-bit shapes";
}

// 64-bit keys (e.g. addresses, sizes) sharing low words but differing in the high word must not
// collapse -- the old 32-bit-constant combiner under-mixed the high half.
TEST(HashCollisionTest, DistinguishesHighWordOf64BitValues) {
    EXPECT_NE(
        hash_objects_with_default_seed(uint64_t{0x0000'0001'0000'0000ULL}),
        hash_objects_with_default_seed(uint64_t{0x0000'0002'0000'0000ULL}));
    EXPECT_NE(
        hash_objects_with_default_seed(uint64_t{1}), hash_objects_with_default_seed(uint64_t{1} << 32));
}

// Avalanche: flipping a single input bit should flip ~half the 64 output bits. A weak mixer
// leaves long runs of output bits untouched. We require every single-bit input flip to change
// at least a quarter of the output bits (16/64) -- a loose bound a good mixer clears easily and
// a shift-and-add combiner fails on the high input bits.
TEST(HashCollisionTest, SingleBitFlipAvalanche) {
    const uint64_t base = 0x0123456789abcdefULL;
    const hash_t base_hash = hash_objects_with_default_seed(base);
    int worst = 64;
    for (int bit = 0; bit < 64; ++bit) {
        const uint64_t flipped = base ^ (uint64_t{1} << bit);
        const hash_t h = hash_objects_with_default_seed(flipped);
        const int changed = __builtin_popcountll(h ^ base_hash);
        worst = std::min(worst, changed);
        EXPECT_GE(changed, 16) << "weak avalanche when flipping input bit " << bit;
    }
    // Sanity: the typical case should be much closer to the ideal of 32.
    EXPECT_GE(worst, 16);
}

// The mixer must spread bits across the whole 64-bit space, not cluster small keys in a corner.
// Hash a run of consecutive small integers and require the OR of all results to light up every
// bit -- a combiner that only touches low bits (the old one, for small inputs) fails this.
TEST(HashCollisionTest, SmallKeysCoverAllOutputBits) {
    hash_t or_all = 0;
    hash_t and_all = ~hash_t{0};
    for (uint32_t i = 0; i < 4096; ++i) {
        const hash_t h = hash_objects_with_default_seed(i);
        or_all |= h;
        and_all &= h;
    }
    EXPECT_EQ(or_all, ~hash_t{0}) << "some output bit is never set across small keys";
    EXPECT_EQ(and_all, hash_t{0}) << "some output bit is always set across small keys";
}

}  // namespace
}  // namespace ttsl::hash
