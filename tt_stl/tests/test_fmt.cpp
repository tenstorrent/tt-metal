// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for tt_stl/fmt.hpp - lightweight formatters
//
// This test verifies that fmt.hpp provides working formatters for common types
// without requiring the heavy dependencies (reflect, nlohmann/json) that
// reflection.hpp pulls in.

#include <gmock/gmock.h>
#include <tt_stl/fmt.hpp>

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

// Test enum (should work via enchantum)
enum class TestEnum { Value1, Value2, Value3 };

// Mock pointer type for testing non-void pointer formatting
struct MockDevice {
    int id;
};

namespace {

using ::testing::HasSubstr;

// Test std::vector<int> formatting
TEST(FmtTest, VectorInt) {
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::string result = fmt::format("{}", vec);
    EXPECT_THAT(result, HasSubstr("1"));
    EXPECT_THAT(result, HasSubstr("2"));
    EXPECT_THAT(result, HasSubstr("3"));
}

// Test std::optional<T> formatting
TEST(FmtTest, Optional) {
    std::optional<int> opt_val = 42;
    std::string result1 = fmt::format("{}", opt_val);
    EXPECT_THAT(result1, HasSubstr("42"));

    std::optional<int> opt_empty = std::nullopt;
    std::string result2 = fmt::format("{}", opt_empty);
    // Should format as nullopt or None or similar
    EXPECT_TRUE(
        result2.find("nullopt") != std::string::npos || result2.find("None") != std::string::npos ||
        result2.find("null") != std::string::npos);
}

// Test std::optional<const T> (edge case mentioned in fmt.hpp comments)
TEST(FmtTest, OptionalConst) {
    std::optional<const bool> opt_const = true;
    std::string result = fmt::format("{}", opt_const);
    EXPECT_TRUE(result.find("true") != std::string::npos || result.find('1') != std::string::npos);
}

// Test container of non-void pointers (critical edge case)
TEST(FmtTest, VectorOfPointers) {
    MockDevice d1{1}, d2{2}, d3{3};
    std::vector<MockDevice*> ptrs = {&d1, &d2, &d3};
    std::string result = fmt::format("{}", ptrs);
    // Should format as pointers (void* representation)
    EXPECT_FALSE(result.empty());
    // Should contain some indication of pointer values
    EXPECT_TRUE(result.find("0x") != std::string::npos || result.find("ptr") != std::string::npos);
}

// Test std::array
TEST(FmtTest, Array) {
    std::array<int, 3> arr = {10, 20, 30};
    std::string result = fmt::format("{}", arr);
    EXPECT_THAT(result, HasSubstr("10"));
    EXPECT_THAT(result, HasSubstr("20"));
    EXPECT_THAT(result, HasSubstr("30"));
}

// Test std::map
TEST(FmtTest, Map) {
    std::map<std::string, int> m = {{"a", 1}, {"b", 2}, {"c", 3}};
    std::string result = fmt::format("{}", m);
    EXPECT_TRUE(result.find('a') != std::string::npos || result.find("\"a\"") != std::string::npos);
    EXPECT_THAT(result, HasSubstr("1"));
}

// Test std::set
TEST(FmtTest, Set) {
    std::set<int> s = {3, 1, 4, 1, 5};
    std::string result = fmt::format("{}", s);
    EXPECT_THAT(result, HasSubstr("1"));
    EXPECT_THAT(result, HasSubstr("3"));
    EXPECT_THAT(result, HasSubstr("4"));
    EXPECT_THAT(result, HasSubstr("5"));
}

// Test std::unordered_map
TEST(FmtTest, UnorderedMap) {
    std::unordered_map<std::string, int> um = {{"x", 10}, {"y", 20}};
    std::string result = fmt::format("{}", um);
    EXPECT_TRUE(result.find('x') != std::string::npos || result.find("\"x\"") != std::string::npos);
    EXPECT_THAT(result, HasSubstr("10"));
}

// Test std::tuple
TEST(FmtTest, Tuple) {
    std::tuple<int, std::string, bool> t = {42, "test", true};
    std::string result = fmt::format("{}", t);
    EXPECT_THAT(result, HasSubstr("42"));
    EXPECT_THAT(result, HasSubstr("test"));
    EXPECT_TRUE(result.find("true") != std::string::npos || result.find('1') != std::string::npos);
}

// Test std::variant
TEST(FmtTest, Variant) {
    std::variant<int, std::string> v1 = 100;
    std::string result1 = fmt::format("{}", v1);
    EXPECT_THAT(result1, HasSubstr("100"));

    std::variant<int, std::string> v2 = std::string("hello");
    std::string result2 = fmt::format("{}", v2);
    EXPECT_THAT(result2, HasSubstr("hello"));
}

// Test enum formatting (via enchantum)
TEST(FmtTest, Enum) {
    TestEnum e = TestEnum::Value2;
    std::string result = fmt::format("{}", e);
    EXPECT_FALSE(result.empty());
    // Should format as enum name, not just number
    EXPECT_TRUE(
        result.find("Value2") != std::string::npos || result.find("value2") != std::string::npos ||
        result.find("Value") != std::string::npos);
}

// Test std::filesystem::path
TEST(FmtTest, FilesystemPath) {
    std::filesystem::path p = "/some/path/to/file.txt";
    std::string result = fmt::format("{}", p);
    EXPECT_TRUE(result.find("file.txt") != std::string::npos || result.find("path") != std::string::npos);
}

// Test ttsl::SmallVector
TEST(FmtTest, SmallVector) {
    ttsl::SmallVector<int, 4> sv = {1, 2, 3};
    std::string result = fmt::format("{}", sv);
    EXPECT_THAT(result, HasSubstr("1"));
    EXPECT_THAT(result, HasSubstr("2"));
    EXPECT_THAT(result, HasSubstr("3"));
}

// Test ttsl::StrongType
TEST(FmtTest, StrongType) {
    using UserId = ttsl::StrongType<int, struct UserIdTag>;
    UserId id{12345};
    std::string result = fmt::format("{}", id);
    EXPECT_THAT(result, HasSubstr("12345"));
}

// Test nested containers
TEST(FmtTest, NestedContainers) {
    std::vector<std::vector<int>> nested = {{1, 2}, {3, 4}, {5}};
    std::string result = fmt::format("{}", nested);
    EXPECT_THAT(result, HasSubstr("1"));
    EXPECT_THAT(result, HasSubstr("2"));
    EXPECT_THAT(result, HasSubstr("3"));
}

// Test empty containers
TEST(FmtTest, EmptyContainers) {
    std::vector<int> empty_vec;
    std::string result = fmt::format("{}", empty_vec);
    EXPECT_FALSE(result.empty());  // Should format something, even if empty

    std::optional<int> empty_opt = std::nullopt;
    std::string result2 = fmt::format("{}", empty_opt);
    EXPECT_FALSE(result2.empty());
}

}  // namespace
