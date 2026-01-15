// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <tt_stl/reflection.hpp>
#include <sstream>
#include <string>

// Define test types in a DIFFERENT namespace to avoid ADL finding the operators
// This is the scenario that triggers the ordering issue
namespace test_types {

// A simple Reflectable struct (uses reflect library)
struct ReflectablePoint {
    int x;
    int y;
};

// A struct with compile-time attributes that contains a Reflectable field
struct ContainerWithReflectable {
    static constexpr auto attribute_names = std::make_tuple("name", "point");
    auto attribute_values() const { return std::make_tuple(name, point); }

    std::string name;
    ReflectablePoint point;
};

// Another Reflectable struct to test nested scenarios
struct ReflectableRect {
    ReflectablePoint top_left;
    ReflectablePoint bottom_right;
};

// Enum for testing enum printing
enum class Color { Red, Green, Blue };

}  // namespace test_types

namespace ttsl::reflection {
namespace {

using test_types::ContainerWithReflectable;
using test_types::ReflectablePoint;
using test_types::ReflectableRect;
using ::testing::HasSubstr;

TEST(ReflectionTest, PrintReflectableStruct) {
    ReflectablePoint p{10, 20};
    std::stringstream ss;
    ss << p;
    std::string result = ss.str();

    EXPECT_THAT(result, HasSubstr("ReflectablePoint"));
    EXPECT_THAT(result, HasSubstr("x=10"));
    EXPECT_THAT(result, HasSubstr("y=20"));
}

TEST(ReflectionTest, PrintNestedReflectableStruct) {
    ReflectableRect rect{{1, 2}, {3, 4}};
    std::stringstream ss;
    ss << rect;
    std::string result = ss.str();

    EXPECT_THAT(result, HasSubstr("ReflectableRect"));
    EXPECT_THAT(result, HasSubstr("top_left="));
    EXPECT_THAT(result, HasSubstr("bottom_right="));
}

TEST(ReflectionTest, PrintStructWithCompileTimeAttributesContainingReflectable) {
    // This test verifies that the template ordering is correct.
    // ContainerWithReflectable uses compile-time attributes (supports_conversion_to_string_v),
    // and contains a ReflectablePoint field that needs the Reflectable operator<<.
    ContainerWithReflectable container{"test", {5, 10}};
    std::stringstream ss;
    ss << container;
    std::string result = ss.str();

    EXPECT_THAT(result, HasSubstr("ContainerWithReflectable"));
    EXPECT_THAT(result, HasSubstr("name=test"));
    EXPECT_THAT(result, HasSubstr("point="));
    // The nested ReflectablePoint should also be printed correctly
    EXPECT_THAT(result, HasSubstr("x=5"));
    EXPECT_THAT(result, HasSubstr("y=10"));
}

TEST(ReflectionTest, PrintVectorOfReflectable) {
    std::vector<ReflectablePoint> points{{1, 2}, {3, 4}};
    std::stringstream ss;
    ss << points;
    std::string result = ss.str();

    EXPECT_THAT(result, HasSubstr("x=1"));
    EXPECT_THAT(result, HasSubstr("y=2"));
    EXPECT_THAT(result, HasSubstr("x=3"));
    EXPECT_THAT(result, HasSubstr("y=4"));
}

TEST(ReflectionTest, PrintOptionalReflectable) {
    std::optional<ReflectablePoint> opt_point = ReflectablePoint{7, 8};
    std::stringstream ss;
    ss << opt_point;
    std::string result = ss.str();

    EXPECT_THAT(result, HasSubstr("x=7"));
    EXPECT_THAT(result, HasSubstr("y=8"));
}

TEST(ReflectionTest, PrintOptionalNullopt) {
    std::optional<ReflectablePoint> opt_point = std::nullopt;
    std::stringstream ss;
    ss << opt_point;
    EXPECT_EQ(ss.str(), "std::nullopt");
}

TEST(ReflectionTest, FmtFormatReflectable) {
    ReflectablePoint p{42, 99};
    std::string result = fmt::format("{}", p);

    EXPECT_THAT(result, HasSubstr("ReflectablePoint"));
    EXPECT_THAT(result, HasSubstr("x=42"));
    EXPECT_THAT(result, HasSubstr("y=99"));
}

TEST(ReflectionTest, FmtFormatContainerWithReflectable) {
    ContainerWithReflectable container{"fmt_test", {15, 25}};
    std::string result = fmt::format("{}", container);

    EXPECT_THAT(result, HasSubstr("ContainerWithReflectable"));
    EXPECT_THAT(result, HasSubstr("name=fmt_test"));
    EXPECT_THAT(result, HasSubstr("x=15"));
    EXPECT_THAT(result, HasSubstr("y=25"));
}

}  // namespace
}  // namespace ttsl::reflection
