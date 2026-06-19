// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Intentionally includes only <tt_stl/hash.hpp> (not reflection.hpp) so this also
// verifies the hashing facility is self-contained.
#include <gmock/gmock.h>
#include <tt_stl/hash.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt_stl/small_vector.hpp>

// Test types live in their own namespace, mirroring test_reflection.cpp.
namespace test_types {

// Plain aggregate: hashed via reflection (no opt-in needed).
struct Point {
    int x;
    int y;
};

// Opts in via the compile-time attributes hook.
struct Sized {
    static constexpr auto attribute_names = std::make_tuple("height", "width");
    auto attribute_values() const { return std::forward_as_tuple(height, width); }

    uint32_t height;
    uint32_t width;
};

// Opts in via a to_hash() method; returns a sentinel so we can confirm it was used.
struct WithToHash {
    ttsl::hash::hash_t to_hash() const { return 0xABCDEF; }
};

// Has BOTH a to_hash() method and attributes: to_hash() must win.
struct ToHashBeatsAttributes {
    static constexpr auto attribute_names = std::make_tuple("v");
    auto attribute_values() const { return std::forward_as_tuple(v); }
    ttsl::hash::hash_t to_hash() const { return 0x111111; }

    int v;
};

// Gets a std::hash specialization (below) AND has a to_hash() method: std::hash must win.
struct StdHashBeatsToHash {
    ttsl::hash::hash_t to_hash() const { return 0x222222; }
};

}  // namespace test_types

template <>
struct std::hash<test_types::StdHashBeatsToHash> {
    std::size_t operator()(const test_types::StdHashBeatsToHash& /*unused*/) const noexcept { return 0x999999; }
};

namespace ttsl::hash {
namespace {

using test_types::Point;
using test_types::Sized;
using test_types::StdHashBeatsToHash;
using test_types::ToHashBeatsAttributes;
using test_types::WithToHash;

// ---- Seeding / determinism --------------------------------------------------

TEST(HashTest, IsDeterministic) {
    EXPECT_EQ(hash_objects_with_default_seed(42), hash_objects_with_default_seed(42));
    EXPECT_EQ(hash_objects_with_default_seed(std::string{"abc"}), hash_objects_with_default_seed(std::string{"abc"}));
}

TEST(HashTest, DistinctValuesGiveDistinctHashes) {
    EXPECT_NE(hash_objects_with_default_seed(1), hash_objects_with_default_seed(2));
    EXPECT_NE(hash_objects_with_default_seed(std::string{"abc"}), hash_objects_with_default_seed(std::string{"abd"}));
}

TEST(HashTest, DefaultSeedMatchesExplicitSeed) {
    EXPECT_EQ(hash_objects_with_default_seed(7), hash_objects(DEFAULT_SEED, 7));
}

TEST(HashTest, SeedAffectsResult) { EXPECT_NE(hash_objects(1, 7), hash_objects(2, 7)); }

// ---- Scalar / std::hash dispatch --------------------------------------------

TEST(HashTest, IntegerHashesToItself) { EXPECT_EQ(detail::hash_object(12345), 12345ULL); }

TEST(HashTest, StdHashableTypeUsesStdHash) {
    const std::string s = "hello";
    EXPECT_EQ(detail::hash_object(s), std::hash<std::string>{}(s));
}

// ---- Recursive / structural composition -------------------------------------

TEST(HashTest, VectorDependsOnElements) {
    EXPECT_EQ(
        hash_objects_with_default_seed(std::vector<int>{1, 2, 3}),
        hash_objects_with_default_seed(std::vector<int>{1, 2, 3}));
    EXPECT_NE(
        hash_objects_with_default_seed(std::vector<int>{1, 2, 3}),
        hash_objects_with_default_seed(std::vector<int>{1, 2, 4}));
}

TEST(HashTest, VectorIsOrderSensitive) {
    EXPECT_NE(
        hash_objects_with_default_seed(std::vector<int>{1, 2}), hash_objects_with_default_seed(std::vector<int>{2, 1}));
}

TEST(HashTest, NestedContainersHash) {
    const std::vector<std::vector<int>> a{{1, 2}, {3}};
    const std::vector<std::vector<int>> b{{1, 2}, {3}};
    const std::vector<std::vector<int>> c{{1}, {2, 3}};
    EXPECT_EQ(hash_objects_with_default_seed(a), hash_objects_with_default_seed(b));
    EXPECT_NE(hash_objects_with_default_seed(a), hash_objects_with_default_seed(c));
}

TEST(HashTest, TupleAndPair) {
    EXPECT_EQ(
        hash_objects_with_default_seed(std::make_tuple(1, std::string{"a"})),
        hash_objects_with_default_seed(std::make_tuple(1, std::string{"a"})));
    EXPECT_NE(
        hash_objects_with_default_seed(std::make_pair(1, 2)), hash_objects_with_default_seed(std::make_pair(2, 1)));
}

TEST(HashTest, ArrayHashesElements) {
    EXPECT_EQ(
        hash_objects_with_default_seed(std::array<int, 3>{1, 2, 3}),
        hash_objects_with_default_seed(std::array<int, 3>{1, 2, 3}));
    EXPECT_NE(
        hash_objects_with_default_seed(std::array<int, 3>{1, 2, 3}),
        hash_objects_with_default_seed(std::array<int, 3>{1, 2, 9}));
}

TEST(HashTest, SpanHashesElements) {
    const std::vector<int> v{4, 5, 6};
    const std::vector<int> w{4, 5, 7};
    EXPECT_EQ(detail::hash_object(std::span<const int>{v}), detail::hash_object(std::span<const int>{v}));
    EXPECT_NE(detail::hash_object(std::span<const int>{v}), detail::hash_object(std::span<const int>{w}));
}

// ---- Container-specific behaviors --------------------------------------------

TEST(HashTest, UnorderedMapIsOrderInvariant) {
    std::unordered_map<int, std::string> a;
    a[1] = "one";
    a[2] = "two";
    a[3] = "three";
    std::unordered_map<int, std::string> b;
    b[3] = "three";
    b[1] = "one";
    b[2] = "two";
    EXPECT_EQ(hash_objects_with_default_seed(a), hash_objects_with_default_seed(b));
}

TEST(HashTest, MapDependsOnContents) {
    const std::map<int, int> a{{1, 10}, {2, 20}};
    const std::map<int, int> b{{1, 10}, {2, 21}};
    EXPECT_NE(hash_objects_with_default_seed(a), hash_objects_with_default_seed(b));
}

// std::optional<int> is itself std::hashable, so to reach the recursive optional
// branch the value type must NOT be std::hashable (an aggregate here).
TEST(HashTest, EmptyOptionalHashesToZero) { EXPECT_EQ(detail::hash_object(std::optional<Point>{}), 0ULL); }

TEST(HashTest, OptionalHashesItsValue) {
    EXPECT_EQ(detail::hash_object(std::optional<Point>{Point{1, 2}}), detail::hash_object(Point{1, 2}));
}

// Same numeric value held in different alternatives must hash differently.
TEST(HashTest, VariantIsIndexSensitive) {
    const std::variant<int, long> a = 5;   // index 0
    const std::variant<int, long> b = 5L;  // index 1
    EXPECT_NE(hash_objects_with_default_seed(a), hash_objects_with_default_seed(b));
}

TEST(HashTest, SmallVectorIsHashable) {
    const ttsl::SmallVector<int> a{1, 2, 3};
    const ttsl::SmallVector<int> b{1, 2, 3};
    const ttsl::SmallVector<int> c{1, 2, 4};
    EXPECT_EQ(hash_objects_with_default_seed(a), hash_objects_with_default_seed(b));
    EXPECT_NE(hash_objects_with_default_seed(a), hash_objects_with_default_seed(c));
}

// ---- Customization points & their priority ----------------------------------

TEST(HashTest, AggregateStructHashedByReflection) {
    EXPECT_EQ(hash_objects_with_default_seed(Point{1, 2}), hash_objects_with_default_seed(Point{1, 2}));
    EXPECT_NE(hash_objects_with_default_seed(Point{1, 2}), hash_objects_with_default_seed(Point{1, 3}));
}

TEST(HashTest, CompileTimeAttributesHashed) {
    EXPECT_EQ(hash_objects_with_default_seed(Sized{4, 8}), hash_objects_with_default_seed(Sized{4, 8}));
    EXPECT_NE(hash_objects_with_default_seed(Sized{4, 8}), hash_objects_with_default_seed(Sized{8, 4}));
}

TEST(HashTest, ToHashMethodIsUsed) { EXPECT_EQ(detail::hash_object(WithToHash{}), 0xABCDEFULL); }

TEST(HashTest, ToHashBeatsCompileTimeAttributes) {
    EXPECT_EQ(detail::hash_object(ToHashBeatsAttributes{42}), 0x111111ULL);
}

TEST(HashTest, StdHashBeatsToHash) { EXPECT_EQ(detail::hash_object(StdHashBeatsToHash{}), 0x999999ULL); }

// ---- hash_combine -----------------------------------------------------------

TEST(HashTest, HashCombineIsDeterministic) {
    std::size_t s1 = 0;
    std::size_t s2 = 0;
    hash_combine(s1, 123);
    hash_combine(s2, 123);
    EXPECT_EQ(s1, s2);
}

TEST(HashTest, HashCombineDependsOnValue) {
    std::size_t s1 = 0;
    std::size_t s2 = 0;
    hash_combine(s1, 123);
    hash_combine(s2, 124);
    EXPECT_NE(s1, s2);
}

}  // namespace
}  // namespace ttsl::hash
