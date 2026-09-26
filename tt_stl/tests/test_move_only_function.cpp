// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/move_only_function.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace ttsl {
namespace {

// Asserted at build time so a change to the backing type cannot quietly regress them.
static_assert(!std::is_copy_constructible_v<move_only_function<void()>>, "must be move-only");
static_assert(!std::is_copy_assignable_v<move_only_function<void()>>, "must be move-only");
// A non-const lvalue is the case a greedy forwarding constructor would swallow.
static_assert(
    !std::is_constructible_v<move_only_function<void()>, move_only_function<void()>&>,
    "must not be constructible from a non-const lvalue of itself");
static_assert(std::is_nothrow_move_constructible_v<move_only_function<void()>>, "move must be noexcept");
static_assert(
    sizeof(move_only_function<void()>) <= sizeof(std::function<void()>),
    "must not be larger than the std::function it replaces");
static_assert(!std::is_convertible_v<move_only_function<void()>, int>, "operator bool must be explicit");
static_assert(std::is_invocable_v<move_only_function<void()>&>, "must be invocable with its own signature");
static_assert(
    !std::is_invocable_v<move_only_function<void()>&, int, int>,
    "must not claim to be invocable with a wrong argument list");

TEST(MoveOnlyFunctionTest, CPU_InvokesLambda) {
    int calls = 0;
    move_only_function<void()> f{[&calls]() { ++calls; }};
    f();
    f();
    EXPECT_EQ(calls, 2);
}

TEST(MoveOnlyFunctionTest, CPU_ReturnsValueAndTakesArguments) {
    move_only_function<std::uint64_t(std::uint64_t, std::uint64_t)> add{
        [](std::uint64_t a, std::uint64_t b) { return a + b; }};
    EXPECT_EQ(add(2, 3), 5u);
}

// The reason this type exists: std::function cannot store a move-only capture at all.
TEST(MoveOnlyFunctionTest, CPU_HoldsMoveOnlyCapture) {
    auto owned = std::make_unique<int>(7);
    move_only_function<int()> f{[owned = std::move(owned)]() { return *owned; }};
    EXPECT_EQ(f(), 7);
}

TEST(MoveOnlyFunctionTest, CPU_EmptyByDefault) {
    move_only_function<void()> f;
    EXPECT_FALSE(static_cast<bool>(f));
    EXPECT_TRUE(f == nullptr);
    EXPECT_FALSE(static_cast<bool>(f));
}

TEST(MoveOnlyFunctionTest, CPU_EngagedReportsNonEmpty) {
    // Captureless, hence trivially destructible: the case that misreports as empty if the RTTI
    // affordance is ever dropped (#57444).
    move_only_function<void()> f{[]() {}};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_FALSE(f == nullptr);
    EXPECT_TRUE(static_cast<bool>(f));
}

TEST(MoveOnlyFunctionTest, CPU_MoveConstructionTransfersTarget) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst{std::move(src)};
    EXPECT_TRUE(static_cast<bool>(dst));
    EXPECT_EQ(dst(), 42);
}

TEST(MoveOnlyFunctionTest, CPU_MoveAssignmentReplacesTarget) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst{[]() { return 1; }};
    dst = std::move(src);
    EXPECT_EQ(dst(), 42);
}

// Exceeds the inline buffer, so this is the heap path.
TEST(MoveOnlyFunctionTest, CPU_HoldsCaptureLargerThanInlineBuffer) {
    struct Big {
        std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    };
    static_assert(sizeof(Big) > detail::kMoveOnlyFunctionInlinePointers * sizeof(void*));
    Big big{};
    move_only_function<std::uint64_t()> f{[big]() { return big.data[7]; }};
    EXPECT_EQ(f(), 8u);
}

// std::function and std::move_only_function both yield an empty wrapper here. The backing type
// would store the null pointer as an ordinary target, report engaged, and then call through it.
TEST(MoveOnlyFunctionTest, CPU_NullFunctionPointerIsEmpty) {
    void (*fp)() = nullptr;
    move_only_function<void()> f{fp};
    EXPECT_FALSE(static_cast<bool>(f));
    EXPECT_TRUE(f == nullptr);
    EXPECT_THROW(f(), std::bad_function_call);
}

TEST(MoveOnlyFunctionTest, CPU_NullptrConstructsEmpty) {
    move_only_function<void()> f{nullptr};
    EXPECT_FALSE(static_cast<bool>(f));
}

TEST(MoveOnlyFunctionTest, CPU_NonNullFunctionPointerIsEngaged) {
    move_only_function<int()> f{+[]() { return 5; }};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_EQ(f(), 5);
}

// The backing type leaves a moved-from wrapper engaged, so operator bool lies and calling it
// dereferences null.
TEST(MoveOnlyFunctionTest, CPU_MovedFromIsEmpty) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst{std::move(src)};
    // Inspecting the moved-from source is the point of the test: its emptiness is the contract.
    EXPECT_FALSE(static_cast<bool>(src));  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(src == nullptr);           // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(static_cast<bool>(dst));
}

TEST(MoveOnlyFunctionTest, CPU_MoveAssignLeavesSourceEmpty) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst;
    dst = std::move(src);
    EXPECT_FALSE(static_cast<bool>(src));  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(dst(), 42);
}

// Heap-stored targets corrupt on self-move without the guard in operator=.
TEST(MoveOnlyFunctionTest, CPU_SelfMoveAssignIsSafe) {
    struct Big {
        std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    };
    Big big{};
    move_only_function<std::uint64_t()> f{[big]() { return big.data[0]; }};
    auto& alias = f;
    f = std::move(alias);
    ASSERT_TRUE(static_cast<bool>(f));
    EXPECT_EQ(f(), 1u);
}

// std::move_only_function yields an empty wrapper here; the backing type would store it engaged.
TEST(MoveOnlyFunctionTest, CPU_EmptyStdFunctionIsEmpty) {
    std::function<void()> empty;
    move_only_function<void()> f{empty};
    EXPECT_FALSE(static_cast<bool>(f));
}

TEST(MoveOnlyFunctionTest, CPU_NonEmptyStdFunctionIsEngaged) {
    std::function<int()> src{[]() { return 3; }};
    move_only_function<int()> f{src};
    ASSERT_TRUE(static_cast<bool>(f));
    EXPECT_EQ(f(), 3);
}

// Pinned deliberately: std::move_only_function makes this undefined, so at the C++23 switch this
// test failing is the intended signal.
TEST(MoveOnlyFunctionTest, CPU_CallingEmptyThrows) {
    move_only_function<void()> f;
    EXPECT_THROW(f(), std::bad_function_call);
}

}  // namespace
}  // namespace ttsl
