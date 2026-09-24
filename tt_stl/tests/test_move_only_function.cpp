// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/move_only_function.hpp>

#include <cstdint>
#include <array>
#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ttsl {
namespace {

// Asserted at build time so a change to the backing type cannot quietly regress them.
static_assert(!std::is_copy_constructible_v<move_only_function<void()>>, "must be move-only");
static_assert(!std::is_copy_assignable_v<move_only_function<void()>>, "must be move-only");
static_assert(std::is_nothrow_move_constructible_v<move_only_function<void()>>, "move must be noexcept");
static_assert(
    sizeof(move_only_function<void()>) <= sizeof(std::function<void()>),
    "must not be larger than the std::function it replaces");

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
    EXPECT_FALSE(f.has_value());
}

TEST(MoveOnlyFunctionTest, CPU_EngagedReportsNonEmpty) {
    // Captureless, hence trivially destructible: the case that misreports as empty if the RTTI
    // affordance is ever dropped (#57444).
    move_only_function<void()> f{[]() {}};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_FALSE(f == nullptr);
    EXPECT_TRUE(f.has_value());
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

// Pinned deliberately: std::move_only_function makes this undefined, so at the C++23 switch this
// test failing is the intended signal.
TEST(MoveOnlyFunctionTest, CPU_CallingEmptyThrows) {
    move_only_function<void()> f;
    EXPECT_THROW(f(), std::bad_function_call);
}

struct CallableCounts {
    int moves = 0;
    int destructors = 0;
    int deleted = 0;
};

struct EmptyCallablePadding {};

// This owning target deliberately retains its pointer bytes after destruction, making a
// destroy-then-move self-assignment detectable as a double-free under ASan.
template <typename Padding>
struct OwningCallable {
    int* value;
    CallableCounts* counts;
    [[no_unique_address]] Padding padding{};

    explicit OwningCallable(CallableCounts& counters) : value(new int(7)), counts(&counters) {}
    OwningCallable(const OwningCallable&) = delete;
    OwningCallable(OwningCallable&& other) noexcept : value(std::exchange(other.value, nullptr)), counts(other.counts) {
        ++counts->moves;
    }
    ~OwningCallable() {
        ++counts->destructors;
        if (value) {
            ++counts->deleted;
            delete value;
        }
    }
    int operator()() const { return *value; }
};

using InlineOwningCallable = OwningCallable<EmptyCallablePadding>;
using HeapOwningCallable = OwningCallable<std::array<std::byte, 64>>;
static_assert(sizeof(InlineOwningCallable) <= detail::kMoveOnlyFunctionInlinePointers * sizeof(void*));
static_assert(sizeof(HeapOwningCallable) > detail::kMoveOnlyFunctionInlinePointers * sizeof(void*));

template <typename Target>
class MoveOnlyFunctionStorageTest : public ::testing::Test {};

using MoveOnlyFunctionTargets = ::testing::Types<InlineOwningCallable, HeapOwningCallable>;
TYPED_TEST_SUITE(MoveOnlyFunctionStorageTest, MoveOnlyFunctionTargets);

TYPED_TEST(MoveOnlyFunctionStorageTest, CPU_MoveConstructionEmptiesSource) {
    CallableCounts counts;
    {
        move_only_function<int()> source{TypeParam{counts}};
        move_only_function<int()> destination{std::move(source)};
        EXPECT_EQ(destination(), 7);
        EXPECT_FALSE(static_cast<bool>(source));
        EXPECT_FALSE(source.has_value());
        EXPECT_EQ(source, nullptr);
        EXPECT_THROW(source(), std::bad_function_call);
        EXPECT_EQ(counts.deleted, 0);
    }
    EXPECT_EQ(counts.deleted, 1);
}

TYPED_TEST(MoveOnlyFunctionStorageTest, CPU_MoveAssignmentEmptiesSource) {
    CallableCounts source_counts;
    CallableCounts destination_counts;
    {
        move_only_function<int()> source{TypeParam{source_counts}};
        move_only_function<int()> destination{TypeParam{destination_counts}};
        destination = std::move(source);
        EXPECT_EQ(destination(), 7);
        EXPECT_FALSE(static_cast<bool>(source));
        EXPECT_FALSE(source.has_value());
        EXPECT_THROW(source(), std::bad_function_call);
        EXPECT_EQ(destination_counts.deleted, 1);
        EXPECT_EQ(source_counts.deleted, 0);
    }
    EXPECT_EQ(source_counts.deleted, 1);
    EXPECT_EQ(destination_counts.deleted, 1);
}

TYPED_TEST(MoveOnlyFunctionStorageTest, CPU_SelfMovePreservesOwnership) {
    CallableCounts counts;
    {
        move_only_function<int()> f{TypeParam{counts}};
        auto& alias = f;
        f = std::move(alias);
        EXPECT_EQ(counts.deleted, 0);
        EXPECT_TRUE(static_cast<bool>(f));
        EXPECT_EQ(f(), 7);
    }
    EXPECT_EQ(counts.deleted, 1);
}

TYPED_TEST(MoveOnlyFunctionStorageTest, CPU_ConstructsTargetWithOneMove) {
    CallableCounts counts;
    {
        TypeParam target{counts};
        move_only_function<int()> f{std::move(target)};
        EXPECT_EQ(counts.moves, 1);
        EXPECT_EQ(counts.destructors, 0);
        EXPECT_EQ(f(), 7);
    }
    EXPECT_EQ(counts.deleted, 1);
}

TYPED_TEST(MoveOnlyFunctionStorageTest, CPU_ResetAndSwapPreserveOwnership) {
    CallableCounts counts;
    {
        move_only_function<int()> f{TypeParam{counts}};
        move_only_function<int()> empty;
        using std::swap;
        swap(f, empty);
        EXPECT_FALSE(static_cast<bool>(f));
        EXPECT_THROW(f(), std::bad_function_call);
        EXPECT_EQ(empty(), 7);
        empty.swap(empty);
        EXPECT_EQ(empty(), 7);
        empty = nullptr;
        EXPECT_FALSE(empty.has_value());
        EXPECT_THROW(empty(), std::bad_function_call);
    }
    EXPECT_EQ(counts.deleted, 1);
}

TEST(MoveOnlyFunctionTest, CPU_EmptyMovesAndSelfMoves) {
    move_only_function<int()> source;
    move_only_function<int()> destination{std::move(source)};
    EXPECT_FALSE(destination.has_value());
    EXPECT_FALSE(source.has_value());
    destination = [] { return 7; };
    destination = std::move(source);
    EXPECT_FALSE(destination.has_value());
    auto& alias = destination;
    destination = std::move(alias);
    EXPECT_FALSE(destination.has_value());
    EXPECT_THROW(destination(), std::bad_function_call);
}

TEST(MoveOnlyFunctionTest, CPU_BoolConversionIsExplicit) {
    using Fn = move_only_function<void()>;
    EXPECT_FALSE((std::is_convertible_v<Fn, bool>));
    EXPECT_FALSE((std::is_convertible_v<Fn, int>));
    EXPECT_TRUE((std::is_constructible_v<bool, const Fn&>));
    Fn f{[] {}};
    EXPECT_TRUE(f ? true : false);
    EXPECT_FALSE(!f);
}

TEST(MoveOnlyFunctionTest, CPU_InvocationTraitsMatchSignature) {
    using Fn = move_only_function<int(int)>;
    EXPECT_TRUE((std::is_invocable_r_v<int, Fn&, int>));
    EXPECT_TRUE((std::invocable<Fn&, short>));
    EXPECT_FALSE((std::is_invocable_v<Fn&>));
    EXPECT_FALSE((std::is_invocable_v<Fn&, int, int>));
    EXPECT_FALSE((std::is_invocable_v<Fn&, const char*>));
    EXPECT_FALSE((std::invocable<Fn&, const char*>));
}

TEST(MoveOnlyFunctionTest, CPU_EmptyStdFunctionRemainsTarget) {
    // C++23 move_only_function treats std::function as an ordinary target, even when empty.
    std::function<int()> empty;
    move_only_function<int()> f{empty};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_THROW(f(), std::bad_function_call);
}

TEST(MoveOnlyFunctionTest, CPU_EmptyWrapperConversionIsEmpty) {
    move_only_function<int()> source;
    move_only_function<long()> destination{std::move(source)};
    EXPECT_FALSE(destination.has_value());
    EXPECT_THROW(destination(), std::bad_function_call);
}

TEST(MoveOnlyFunctionTest, CPU_ThrowingMoveTargetUsesHeap) {
    struct Target {
        bool* throw_on_move;
        int* moves;
        Target(bool& should_throw, int& count) : throw_on_move(&should_throw), moves(&count) {}
        Target(Target&& other) noexcept(false) : throw_on_move(other.throw_on_move), moves(other.moves) {
            ++*moves;
            if (*throw_on_move) {
                throw std::runtime_error("target move");
            }
        }
        int operator()() const { return 7; }
    };
    static_assert(sizeof(Target) <= detail::kMoveOnlyFunctionInlinePointers * sizeof(void*));
    bool throw_on_move = false;
    int moves = 0;
    move_only_function<int()> source{Target{throw_on_move, moves}};
    EXPECT_EQ(moves, 1);
    throw_on_move = true;
    move_only_function<int()> destination{std::move(source)};
    move_only_function<int()> assigned;
    assigned = std::move(destination);
    EXPECT_EQ(moves, 1);
    EXPECT_EQ(assigned(), 7);
    EXPECT_THROW((move_only_function<int()>{Target{throw_on_move, moves}}), std::runtime_error);
}

TEST(MoveOnlyFunctionTest, CPU_InvokesMemberPointers) {
    struct Object {
        int value = 7;
        int method(int increment) const { return value + increment; }
    };
    Object object;
    move_only_function<int(Object&, int)> method{&Object::method};
    EXPECT_EQ(method(object, 2), 9);
    move_only_function<int&(Object&)> data{&Object::value};
    data(object) = 11;
    EXPECT_EQ(object.value, 11);

    decltype(&Object::method) null_method = nullptr;
    decltype(&Object::value) null_data = nullptr;
    move_only_function<int(Object&, int)> empty_method{null_method};
    move_only_function<int&(Object&)> empty_data{null_data};
    EXPECT_FALSE(empty_method.has_value());
    EXPECT_FALSE(empty_data.has_value());
    EXPECT_THROW(empty_method(object, 1), std::bad_function_call);
    EXPECT_THROW(empty_data(object), std::bad_function_call);
}

TEST(MoveOnlyFunctionTest, CPU_ForwardsArgumentsAndDiscardsVoidResult) {
    move_only_function<int(std::unique_ptr<int>)> take{[](std::unique_ptr<int> value) { return *value; }};
    EXPECT_EQ(take(std::make_unique<int>(7)), 7);
    int value = 0;
    move_only_function<void(int&)> update{[](int& target) { return ++target; }};
    update(value);
    EXPECT_EQ(value, 1);
    const move_only_function<int()> mutable_target{[calls = 0]() mutable { return ++calls; }};
    EXPECT_EQ(mutable_target(), 1);
    EXPECT_EQ(mutable_target(), 2);
}

TEST(MoveOnlyFunctionTest, CPU_WrapperConversionInvokesTarget) {
    move_only_function<int()> source{[value = std::make_unique<int>(7)] { return *value; }};
    move_only_function<long()> destination{std::move(source)};
    EXPECT_FALSE(source.has_value());
    EXPECT_EQ(destination(), 7);
}

}  // namespace
}  // namespace ttsl
