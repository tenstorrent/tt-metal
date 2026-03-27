// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/reflection.hpp>
#include <array>
#include <optional>
#include <tuple>
#include <vector>

namespace test_types {

struct TargetType {
    int value;
    bool operator==(const TargetType&) const = default;
};

// Reflectable struct containing only TargetType members
struct PairOfTargets {
    TargetType first;
    TargetType second;
};

// Reflectable struct with nested TargetType in containers
struct ContainerHolder {
    std::vector<TargetType> targets;
    std::optional<TargetType> maybe_target;
};

// Deeply nested Reflectable struct
struct NestedHolder {
    PairOfTargets pair;
    TargetType single;
};

}  // namespace test_types

namespace ttsl::reflection {
namespace {

using test_types::TargetType;

// =============================================================================
// update_object_of_type tests
// =============================================================================

TEST(UpdateObjectOfTypeTest, DirectTypeMatch) {
    TargetType target{42};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 2; }, target);

    EXPECT_EQ(target.value, 84);
}

TEST(UpdateObjectOfTypeTest, NonMatchingTypeThrows) {
    int non_target = 123;
    EXPECT_THROW(update_object_of_type<TargetType>([](TargetType&) {}, non_target), std::runtime_error);
}

TEST(UpdateObjectOfTypeTest, OptionalWithValue) {
    std::optional<TargetType> opt_target = TargetType{50};

    update_object_of_type<TargetType>([](TargetType& t) { t.value += 10; }, opt_target);

    ASSERT_TRUE(opt_target.has_value());
    EXPECT_EQ(opt_target->value, 60);
}

TEST(UpdateObjectOfTypeTest, OptionalWithoutValue) {
    std::optional<TargetType> opt_target = std::nullopt;
    bool callback_called = false;

    update_object_of_type<TargetType>([&](TargetType&) { callback_called = true; }, opt_target);

    EXPECT_FALSE(callback_called);
    EXPECT_FALSE(opt_target.has_value());
}

TEST(UpdateObjectOfTypeTest, VectorOfTargetType) {
    std::vector<TargetType> targets{{1}, {2}, {3}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 10; }, targets);

    ASSERT_EQ(targets.size(), 3);
    EXPECT_EQ(targets[0].value, 10);
    EXPECT_EQ(targets[1].value, 20);
    EXPECT_EQ(targets[2].value, 30);
}

TEST(UpdateObjectOfTypeTest, ArrayOfTargetType) {
    std::array<TargetType, 3> targets{{{5}, {10}, {15}}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value += 100; }, targets);

    EXPECT_EQ(targets[0].value, 105);
    EXPECT_EQ(targets[1].value, 110);
    EXPECT_EQ(targets[2].value, 115);
}

TEST(UpdateObjectOfTypeTest, TupleOfTargetTypes) {
    std::tuple<TargetType, TargetType, TargetType> tuple{TargetType{10}, TargetType{20}, TargetType{30}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 2; }, tuple);

    EXPECT_EQ(std::get<0>(tuple).value, 20);
    EXPECT_EQ(std::get<1>(tuple).value, 40);
    EXPECT_EQ(std::get<2>(tuple).value, 60);
}

TEST(UpdateObjectOfTypeTest, NestedOptionalInVector) {
    std::vector<std::optional<TargetType>> targets{TargetType{1}, std::nullopt, TargetType{3}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 100; }, targets);

    ASSERT_EQ(targets.size(), 3);
    ASSERT_TRUE(targets[0].has_value());
    EXPECT_EQ(targets[0]->value, 100);
    EXPECT_FALSE(targets[1].has_value());
    ASSERT_TRUE(targets[2].has_value());
    EXPECT_EQ(targets[2]->value, 300);
}

TEST(UpdateObjectOfTypeTest, NestedVectorInVector) {
    std::vector<std::vector<TargetType>> nested{{{1}, {2}}, {{3}, {4}, {5}}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value += 10; }, nested);

    ASSERT_EQ(nested.size(), 2);
    ASSERT_EQ(nested[0].size(), 2);
    EXPECT_EQ(nested[0][0].value, 11);
    EXPECT_EQ(nested[0][1].value, 12);
    ASSERT_EQ(nested[1].size(), 3);
    EXPECT_EQ(nested[1][0].value, 13);
    EXPECT_EQ(nested[1][1].value, 14);
    EXPECT_EQ(nested[1][2].value, 15);
}

TEST(UpdateObjectOfTypeTest, ArrayOfOptionals) {
    std::array<std::optional<TargetType>, 3> targets{TargetType{10}, std::nullopt, TargetType{30}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value = -t.value; }, targets);

    ASSERT_TRUE(targets[0].has_value());
    EXPECT_EQ(targets[0]->value, -10);
    EXPECT_FALSE(targets[1].has_value());
    ASSERT_TRUE(targets[2].has_value());
    EXPECT_EQ(targets[2]->value, -30);
}

TEST(UpdateObjectOfTypeTest, EmptyVector) {
    std::vector<TargetType> empty_targets;
    bool callback_called = false;

    update_object_of_type<TargetType>([&](TargetType&) { callback_called = true; }, empty_targets);

    EXPECT_FALSE(callback_called);
    EXPECT_TRUE(empty_targets.empty());
}

TEST(UpdateObjectOfTypeTest, CountingCallback) {
    std::vector<TargetType> targets{{1}, {2}, {3}, {4}, {5}};
    int call_count = 0;

    update_object_of_type<TargetType>([&](TargetType&) { ++call_count; }, targets);

    EXPECT_EQ(call_count, 5);
}

TEST(UpdateObjectOfTypeTest, TupleOfVectors) {
    std::tuple<std::vector<TargetType>, std::vector<TargetType>> tuple{
        std::vector<TargetType>{{1}, {2}}, std::vector<TargetType>{{3}, {4}, {5}}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 10; }, tuple);

    const auto& vec0 = std::get<0>(tuple);
    const auto& vec1 = std::get<1>(tuple);
    ASSERT_EQ(vec0.size(), 2);
    EXPECT_EQ(vec0[0].value, 10);
    EXPECT_EQ(vec0[1].value, 20);
    ASSERT_EQ(vec1.size(), 3);
    EXPECT_EQ(vec1[0].value, 30);
    EXPECT_EQ(vec1[1].value, 40);
    EXPECT_EQ(vec1[2].value, 50);
}

// =============================================================================
// Tests for Reflectable structs containing TargetType members
// =============================================================================

using test_types::ContainerHolder;
using test_types::NestedHolder;
using test_types::PairOfTargets;

TEST(UpdateObjectOfTypeTest, ReflectableWithTargetMembers) {
    PairOfTargets pair{TargetType{10}, TargetType{20}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 3; }, pair);

    EXPECT_EQ(pair.first.value, 30);
    EXPECT_EQ(pair.second.value, 60);
}

TEST(UpdateObjectOfTypeTest, ReflectableWithContainerMembers) {
    ContainerHolder holder{.targets = {{1}, {2}, {3}}, .maybe_target = TargetType{100}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value += 10; }, holder);

    ASSERT_EQ(holder.targets.size(), 3);
    EXPECT_EQ(holder.targets[0].value, 11);
    EXPECT_EQ(holder.targets[1].value, 12);
    EXPECT_EQ(holder.targets[2].value, 13);
    ASSERT_TRUE(holder.maybe_target.has_value());
    EXPECT_EQ(holder.maybe_target->value, 110);
}

TEST(UpdateObjectOfTypeTest, ReflectableWithEmptyOptionalMember) {
    ContainerHolder holder{.targets = {{5}}, .maybe_target = std::nullopt};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 2; }, holder);

    ASSERT_EQ(holder.targets.size(), 1);
    EXPECT_EQ(holder.targets[0].value, 10);
    EXPECT_FALSE(holder.maybe_target.has_value());
}

TEST(UpdateObjectOfTypeTest, NestedReflectableStructs) {
    NestedHolder nested{.pair = {TargetType{1}, TargetType{2}}, .single = TargetType{3}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value *= 100; }, nested);

    EXPECT_EQ(nested.pair.first.value, 100);
    EXPECT_EQ(nested.pair.second.value, 200);
    EXPECT_EQ(nested.single.value, 300);
}

TEST(UpdateObjectOfTypeTest, VectorOfReflectable) {
    std::vector<PairOfTargets> pairs{{TargetType{1}, TargetType{2}}, {TargetType{3}, TargetType{4}}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value += 100; }, pairs);

    ASSERT_EQ(pairs.size(), 2);
    EXPECT_EQ(pairs[0].first.value, 101);
    EXPECT_EQ(pairs[0].second.value, 102);
    EXPECT_EQ(pairs[1].first.value, 103);
    EXPECT_EQ(pairs[1].second.value, 104);
}

TEST(UpdateObjectOfTypeTest, OptionalReflectable) {
    std::optional<PairOfTargets> opt_pair = PairOfTargets{TargetType{7}, TargetType{8}};

    update_object_of_type<TargetType>([](TargetType& t) { t.value = -t.value; }, opt_pair);

    ASSERT_TRUE(opt_pair.has_value());
    EXPECT_EQ(opt_pair->first.value, -7);
    EXPECT_EQ(opt_pair->second.value, -8);
}

}  // namespace
}  // namespace ttsl::reflection
