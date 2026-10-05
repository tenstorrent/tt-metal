// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/move_only_function.hpp>

#include <cstdint>
#include <functional>
#include <initializer_list>
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

// The public surface is std::move_only_function's and nothing more.
static_assert(std::is_same_v<move_only_function<int(char)>::result_type, int>);
static_assert(!std::is_invocable_v<const move_only_function<void()>&>, "R(Args...) is not const-callable");
static_assert(!std::is_constructible_v<move_only_function<void()>, int>, "must reject non-callables");
static_assert(!std::is_constructible_v<move_only_function<void()>, void (*)(int)>, "must reject a wrong signature");
static_assert(
    !std::is_convertible_v<move_only_function<void()>&, detail::MoveOnlyFunctionBase<void()>&>,
    "the backing type must not be reachable");
// Concepts, so the private base's inaccessible members read as unsatisfied rather than an error.
template <typename F>
concept ExposesHasValue = requires(F& f) { f.has_value(); };
template <typename F>
concept ExposesReset = requires(F& f) { f.reset(); };
template <typename F>
concept ExposesType = requires(F& f) { f.type(); };
static_assert(!ExposesHasValue<move_only_function<void()>>, "zoo's accessors must not leak");
static_assert(!ExposesReset<move_only_function<void()>>, "zoo's accessors must not leak");
static_assert(!ExposesType<move_only_function<void()>>, "zoo's accessors must not leak");

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

// Emptying only the target, not the invoker, made these run the moved-out callable.
TEST(MoveOnlyFunctionTest, CPU_CallingMovedFromThrows) {
    move_only_function<int()> src{[p = std::make_unique<int>(42)] { return *p; }};
    move_only_function<int()> dst{std::move(src)};
    ASSERT_FALSE(static_cast<bool>(src));         // NOLINT(bugprone-use-after-move)
    EXPECT_THROW(src(), std::bad_function_call);  // NOLINT(bugprone-use-after-move)
}

TEST(MoveOnlyFunctionTest, CPU_CallingMoveAssignedFromThrows) {
    move_only_function<int()> src{[p = std::make_unique<int>(42)] { return *p; }};
    move_only_function<int()> dst;
    dst = std::move(src);
    ASSERT_FALSE(static_cast<bool>(src));         // NOLINT(bugprone-use-after-move)
    EXPECT_THROW(src(), std::bad_function_call);  // NOLINT(bugprone-use-after-move)
}

// Same for a heap-stored target, whose move steals the pointer rather than moving the object.
TEST(MoveOnlyFunctionTest, CPU_CallingMovedFromHeapTargetThrows) {
    std::uint64_t big[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    move_only_function<std::uint64_t()> src{[big]() { return big[0]; }};
    move_only_function<std::uint64_t()> dst{std::move(src)};
    ASSERT_FALSE(static_cast<bool>(src));         // NOLINT(bugprone-use-after-move)
    EXPECT_THROW(src(), std::bad_function_call);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(dst(), 1u);
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

TEST(MoveOnlyFunctionTest, CPU_AssigningNullptrEmpties) {
    move_only_function<int()> f{[]() { return 1; }};
    f = nullptr;
    EXPECT_FALSE(static_cast<bool>(f));
    EXPECT_TRUE(nullptr == f);
}

TEST(MoveOnlyFunctionTest, CPU_AssigningCallableReplacesTarget) {
    move_only_function<int()> f{[]() { return 1; }};
    f = [p = std::make_unique<int>(2)] { return *p; };
    EXPECT_EQ(f(), 2);
}

TEST(MoveOnlyFunctionTest, CPU_SwapExchangesTargets) {
    move_only_function<int()> a{[]() { return 1; }};
    move_only_function<int()> b;
    a.swap(b);
    EXPECT_FALSE(static_cast<bool>(a));
    EXPECT_EQ(b(), 1);
    swap(a, b);
    EXPECT_EQ(a(), 1);
    EXPECT_FALSE(static_cast<bool>(b));
}

TEST(MoveOnlyFunctionTest, CPU_InPlaceConstructsTarget) {
    struct Adder {
        int base;
        int operator()(int x) const { return base + x; }
    };
    move_only_function<int(int)> f{std::in_place_type<Adder>, 40};
    EXPECT_EQ(f(2), 42);

    struct Summer {
        int total = 0;
        Summer(std::initializer_list<int> values, int extra) {
            for (int v : values) {
                total += v;
            }
            total += extra;
        }
        int operator()() const { return total; }
    };
    move_only_function<int()> g{std::in_place_type<Summer>, {1, 2, 3}, 4};
    EXPECT_EQ(g(), 10);
}

// Pinned deliberately: std::move_only_function makes this undefined, so at the C++23 switch this
// test failing is the intended signal.
TEST(MoveOnlyFunctionTest, CPU_CallingEmptyThrows) {
    move_only_function<void()> f;
    EXPECT_THROW(f(), std::bad_function_call);
}

}  // namespace
}  // namespace ttsl
