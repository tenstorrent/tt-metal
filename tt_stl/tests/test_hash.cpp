// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
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

// Allocation-free variants for the large sweeps (a stack std::array, no per-iteration heap).
template <std::size_t N>
hash_t hash_dims(const std::array<uint32_t, N>& dims) {
    return hash_objects_with_default_seed(ttsl::Span<const uint32_t>(dims.data(), dims.size()));
}
template <std::size_t N>
std::string canonical_dims(const std::array<uint32_t, N>& dims) {
    return canonical_key(ttsl::Span<const uint32_t>(dims.data(), dims.size()));
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
                    seen.insert(hash_dims(std::array<uint32_t, 4>{a, b, c, d}));
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
                seen.insert(hash_dims(std::array<uint32_t, 3>{a, b, c}));
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

// =============================================================================================
// canonical_key: the exact, collision-free companion used to resolve hash collisions (option 1).
// =============================================================================================

std::string canonical_shape(const std::vector<uint32_t>& dims) {
    return canonical_key(ttsl::Span<const uint32_t>(dims.data(), dims.size()));
}

TEST(CanonicalKeyTest, DistinguishesShapes) {
    // The shapes that collided under the old 64-bit hash (issue #45821) must be distinct as exact
    // keys regardless of the hash, so a collision can always be resolved to a correct (rebuild) miss.
    EXPECT_NE(canonical_shape({3, 17, 1, 1}), canonical_shape({1, 152, 1, 1}));
    EXPECT_NE(canonical_shape({3, 17}), canonical_shape({1, 152}));
}

TEST(CanonicalKeyTest, EqualForEqualInputs) {
    EXPECT_EQ(canonical_shape({3, 17, 1, 1}), canonical_shape({3, 17, 1, 1}));
    EXPECT_EQ(canonical_key(uint32_t{42}, uint32_t{7}), canonical_key(uint32_t{42}, uint32_t{7}));
}

TEST(CanonicalKeyTest, OrderAndLengthSensitive) {
    EXPECT_NE(canonical_shape({2, 3}), canonical_shape({3, 2}));
    // Length-prefixing prevents [1,2],[3] from encoding the same as [1],[2,3] etc.
    EXPECT_NE(canonical_shape({1, 2, 3}), canonical_shape({1, 2}));
    EXPECT_NE(canonical_shape({1, 1}), canonical_shape({1}));
}

TEST(CanonicalKeyTest, DistinguishesScalarsEnumsOptionals) {
    EXPECT_NE(canonical_key(uint32_t{1}, uint32_t{2}), canonical_key(uint32_t{2}, uint32_t{1}));
    enum class E : int { A, B };
    EXPECT_NE(canonical_key(E::A), canonical_key(E::B));
    EXPECT_NE(canonical_key(std::optional<uint32_t>{}), canonical_key(std::optional<uint32_t>{0}));
    EXPECT_NE(canonical_key(float{1.0f}), canonical_key(float{2.0f}));
    EXPECT_NE(canonical_key(std::string{"ab"}), canonical_key(std::string{"ba"}));
}

// std::vector<bool> is bit-packed: its iterator yields a proxy reference, not bool&, so it needs a
// dedicated encoding branch (the generic vector branch fails to compile).
TEST(CanonicalKeyTest, DistinguishesVectorBool) {
    EXPECT_EQ(
        canonical_key(std::vector<bool>{true, false, false}), canonical_key(std::vector<bool>{true, false, false}));
    EXPECT_NE(
        canonical_key(std::vector<bool>{true, false, false}), canonical_key(std::vector<bool>{false, true, false}));
    EXPECT_NE(canonical_key(std::vector<bool>{true}), canonical_key(std::vector<bool>{true, false}));

    // For each length 1..10, an all-true vector and an all-false vector (opposite at every index)
    // must encode differently.
    std::vector<bool> mixed_left;
    std::vector<bool> mixed_right;
    for (std::size_t len = 1; len <= 10; ++len) {
        const std::vector<bool> all_true(len, true);
        const std::vector<bool> all_false(len, false);
        EXPECT_NE(canonical_key(all_true), canonical_key(all_false)) << "opposite booleans. length=" << len;
        const bool vb = (0 == (len % 2));
        mixed_left.push_back(vb);
        mixed_right.push_back(!vb);
        EXPECT_NE(canonical_key(mixed_left), canonical_key(mixed_right)) << "mixed opposite booleans. length=" << len;
    }
}

TEST(CanonicalKeyTest, ToHashTakesPrecedenceOverReflection) {
    struct ReflectableWithToHash {
        uint32_t reflected_value;
        uint32_t hash_key;

        hash_t to_hash() const { return hash_objects_with_default_seed(hash_key); }
    };

    static_assert(ttsl::concepts::Reflectable<ReflectableWithToHash>);
    static_assert(ttsl::reflection::detail::supports_to_hash_v<ReflectableWithToHash>);

    const ReflectableWithToHash same_hash_left{.reflected_value = 1, .hash_key = 7};
    const ReflectableWithToHash same_hash_right{.reflected_value = 2, .hash_key = 7};
    const ReflectableWithToHash different_hash{.reflected_value = 1, .hash_key = 8};

    EXPECT_EQ(hash_objects_with_default_seed(same_hash_left), hash_objects_with_default_seed(same_hash_right));
    EXPECT_EQ(canonical_key(same_hash_left), canonical_key(same_hash_right));

    EXPECT_NE(hash_objects_with_default_seed(same_hash_left), hash_objects_with_default_seed(different_hash));
    EXPECT_NE(canonical_key(same_hash_left), canonical_key(different_hash));
}

// Coverage over the same adversarial set used for the hash: the exact key must be injective here
// (it is by construction, but pin it so a future encoding change can't silently regress).
TEST(CanonicalKeyTest, NoCollisionsOverSmall4DShapes) {
    constexpr uint32_t kMax = 40;
    std::unordered_set<std::string> seen;
    size_t count = 0;
    for (uint32_t a = 0; a < kMax; ++a) {
        for (uint32_t b = 0; b < kMax; ++b) {
            for (uint32_t c = 0; c < kMax; ++c) {
                for (uint32_t d = 0; d < kMax; ++d) {
                    seen.insert(canonical_dims(std::array<uint32_t, 4>{a, b, c, d}));
                    ++count;
                }
            }
        }
    }
    EXPECT_EQ(seen.size(), count) << "canonical key collision among small 4-D shapes";
}

// =============================================================================================
// The cache contract: a 64-bit hash collision is resolved by exact comparison of the canonical
// key, so two distinct keys with the same hash coexist and each retrieves its OWN entry -- the
// std::unordered_map bucket-walk doing the work. This mirrors
// tt::tt_metal::program_cache::detail::ProgramCacheKey, replicated here to keep the tt_stl test
// free of tt_metal dependencies.
// =============================================================================================

TEST(ProgramCacheKeyTest, HashCollisionResolvedByCanonicalKey) {
    struct Key {
        uint64_t hash;
        std::string canonical;
        bool operator==(const Key& o) const { return hash == o.hash && canonical == o.canonical; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const { return k.hash; }
    };

    std::unordered_map<Key, int, KeyHash> cache;

    // Two DISTINCT shapes forced to the SAME 64-bit hash -- exactly the #45821 collision.
    Key a{42, canonical_shape({3, 17, 1, 1})};
    Key b{42, canonical_shape({1, 152, 1, 1})};
    ASSERT_EQ(a.hash, b.hash);          // same bucket
    ASSERT_NE(a.canonical, b.canonical);

    cache[a] = 1;  // "program" for shape A
    cache[b] = 2;  // "program" for shape B

    EXPECT_EQ(cache.size(), 2u);  // both coexist in one bucket (native chaining)
    EXPECT_EQ(cache.at(a), 1);    // A retrieves A's entry, NOT B's -> the wrong-hit bug is gone
    EXPECT_EQ(cache.at(b), 2);
    EXPECT_TRUE(cache.contains(a));
    EXPECT_TRUE(cache.contains(b));

    // An identical key (same hash + same canonical) is a real hit and collapses onto the entry.
    Key a_again{42, canonical_shape({3, 17, 1, 1})};
    EXPECT_EQ(cache.at(a_again), 1);
    cache[a_again] = 9;
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_EQ(cache.at(a), 9);
}

// Informational: the per-dispatch host cost of option 1 is one canonical_key build (a full key
// traversal + a small string alloc) on top of the hash that already runs every dispatch. Print
// both so the overhead is a number, not a guess. Loose sanity bound only (machine-dependent).
struct CostAttrs {
    uint32_t a, b, c;
    int32_t e;
    float f;
    static constexpr auto attribute_names = std::forward_as_tuple("a", "b", "c", "e", "f");
    auto attribute_values() const { return std::forward_as_tuple(a, b, c, e, f); }
};

TEST(CanonicalKeyTest, RelativeCostVsHash) {
    const CostAttrs attrs{1, 2, 3, 4, 5.0f};
    const std::vector<uint32_t> shape{1, 152, 24, 32};
    constexpr int N = 1'000'000;

    volatile uint64_t hsink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        hsink ^= hash_objects_with_default_seed(uint64_t{777}, attrs, shape);
    }
    auto t1 = std::chrono::steady_clock::now();
    volatile size_t csink = 0;
    for (int i = 0; i < N; ++i) {
        csink += canonical_key(uint64_t{777}, attrs, shape).size();
    }
    auto t2 = std::chrono::steady_clock::now();

    const double hash_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    const double canon_ns = std::chrono::duration<double, std::nano>(t2 - t1).count() / N;
    std::cout << "[ cost     ] hash_objects=" << hash_ns << " ns/op, canonical_key=" << canon_ns
              << " ns/op, ratio=" << (canon_ns / hash_ns) << "x (key size=" << canonical_key(uint64_t{777}, attrs, shape).size()
              << " bytes)\n";
    EXPECT_GT(hash_ns, 0.0);
    EXPECT_GT(canon_ns, 0.0);
}

}  // namespace
}  // namespace ttsl::hash
