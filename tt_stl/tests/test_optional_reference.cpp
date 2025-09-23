// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt_stl/optional_reference.hpp>

namespace ttsl {
namespace {

using ::testing::ElementsAre;

TEST(OptionalReferenceTest, DefaultConstruction) {
    optional_reference<int> ref;
    EXPECT_FALSE(ref.has_value());
    EXPECT_FALSE(ref);
}

TEST(OptionalReferenceTest, NulloptConstruction) {
    optional_reference<int> ref(std::nullopt);
    EXPECT_FALSE(ref.has_value());
    EXPECT_FALSE(ref);
}

TEST(OptionalReferenceTest, LvalueConstruction) {
    int value = 42;
    optional_reference<int> ref(value);
    EXPECT_TRUE(ref.has_value());
    EXPECT_TRUE(ref);
    EXPECT_EQ(&*ref, &value);
    EXPECT_EQ(*ref, 42);
}

TEST(OptionalReferenceTest, ConstLvalueConstruction) {
    const int value = 42;
    optional_reference<const int> ref(value);
    EXPECT_TRUE(ref.has_value());
    EXPECT_EQ(*ref, 42);
}

TEST(OptionalReferenceTest, ImplicitConversionToConst) {
    int value = 42;
    optional_reference<const int> ref(value);
    EXPECT_TRUE(ref.has_value());
    EXPECT_EQ(*ref, 42);

    // Modify original value
    value = 100;
    EXPECT_EQ(*ref, 100);  // Reference should see the change
}

TEST(OptionalReferenceTest, OptionalConstruction) {
    std::optional<int> opt = 42;
    optional_reference<int> ref(opt);
    EXPECT_TRUE(ref.has_value());
    EXPECT_EQ(*ref, 42);

    // Modify through optional
    opt.value() = 100;
    EXPECT_EQ(*ref, 100);
}

TEST(OptionalReferenceTest, ConstOptionalConstruction) {
    const std::optional<int> opt = 42;
    optional_reference<const int> ref(opt);
    EXPECT_TRUE(ref.has_value());
    EXPECT_EQ(*ref, 42);
}

TEST(OptionalReferenceTest, EmptyOptionalConstruction) {
    std::optional<int> opt;
    optional_reference<int> ref(opt);
    EXPECT_FALSE(ref.has_value());
}

TEST(OptionalReferenceTest, CopyConstruction) {
    int value = 42;
    optional_reference<int> ref1(value);
    optional_reference<int> ref2(ref1);

    EXPECT_TRUE(ref2.has_value());
    EXPECT_EQ(&*ref1, &*ref2);
    EXPECT_EQ(*ref2, 42);
}

TEST(OptionalReferenceTest, MoveConstruction) {
    int value = 42;
    optional_reference<int> ref1(value);
    optional_reference<int> ref2(std::move(ref1)); // NOLINT(performance-move-const-arg)

    EXPECT_TRUE(ref2.has_value());
    EXPECT_EQ(*ref2, 42);
    // Note: ref1 should still be valid after move (it's just a pointer copy)
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(ref1.has_value());
}

TEST(OptionalReferenceTest, CopyAssignment) {
    int value1 = 42;
    int value2 = 100;
    optional_reference<int> ref1(value1);
    optional_reference<int> ref2(value2);

    ref2 = ref1;
    EXPECT_EQ(&*ref1, &*ref2);
    EXPECT_EQ(*ref2, 42);
}

TEST(OptionalReferenceTest, Reset) {
    int value = 42;
    optional_reference<int> ref(value);
    EXPECT_TRUE(ref.has_value());

    ref.reset();
    EXPECT_FALSE(ref.has_value());
}

TEST(OptionalReferenceTest, ArrowOperator) {
    struct TestStruct {
        int value = 42;
        int get() const { return value; }
    };

    TestStruct obj;
    optional_reference<TestStruct> ref(obj);
    EXPECT_EQ(ref->value, 42);
    EXPECT_EQ(ref->get(), 42);
}

TEST(OptionalReferenceTest, ValueMethod) {
    int val = 42;
    optional_reference<int> ref(val);
    EXPECT_EQ(ref.value(), 42);

    // Modify through value()
    ref.value() = 100;
    EXPECT_EQ(val, 100);
}

TEST(OptionalReferenceTest, EqualityOperators) {
    int value1 = 42;
    int value2 = 42;

    optional_reference<int> ref1(value1);
    optional_reference<int> ref2(value1);  // Same object
    optional_reference<int> ref3(value2);  // Different object
    optional_reference<int> empty1;
    optional_reference<int> empty2;

    // Same reference
    EXPECT_EQ(ref1, ref2);

    // Different objects (even with same value)
    EXPECT_NE(ref1, ref3);

    // Empty references are equal
    EXPECT_EQ(empty1, empty2);

    // Empty vs non-empty
    EXPECT_NE(ref1, empty1);
}

TEST(OptionalReferenceTest, ModifyThroughReference) {
    int value = 42;
    optional_reference<int> ref(value);

    *ref = 100;
    EXPECT_EQ(value, 100);

    ref.value() = 200;
    EXPECT_EQ(value, 200);
}

TEST(OptionalReferenceTest, ConstCorrectness) {
    int value = 42;
    const optional_reference<int> ref(value);

    // Can still modify through const optional_reference
    *ref = 100;
    EXPECT_EQ(value, 100);
}

TEST(OptionalReferenceTest, UseCaseExample) {
    auto test_function = [](optional_reference<const int> val) {
        if (val) {
            EXPECT_GE(*val, 0);
        }
    };

    test_function(3);

    int a = 5;
    test_function(a);

    test_function(std::nullopt);
}

TEST(OptionalReferenceTest, TemporaryBindingToConst) {
    auto test_function = [](optional_reference<const int> val) { EXPECT_EQ(*val, 42); };
    test_function(42);
    test_function(std::make_optional(42));
}

}  // namespace
}  // namespace ttsl
