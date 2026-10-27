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

static_assert(std::is_same_v<move_only_function<int(char)>::result_type, int>);
static_assert(!std::is_invocable_v<const move_only_function<void()>&>, "R(Args...) is not const-callable");
static_assert(!std::is_constructible_v<move_only_function<void()>, int>, "must reject non-callables");
static_assert(!std::is_constructible_v<move_only_function<void()>, void (*)(int)>, "must reject a wrong signature");
struct LvalueOnlyCallable {
    int operator()() & { return 0; }
};
static_assert(
    !std::is_constructible_v<move_only_function<int()>, LvalueOnlyCallable>, "a target must be rvalue-callable too");
static_assert(
    !std::is_constructible_v<move_only_function<int()>, std::in_place_type_t<LvalueOnlyCallable>>,
    "a target must be rvalue-callable too");
struct ReturnsInt {
    int operator()() const { return 42; }
};
#if __has_builtin(__reference_converts_from_temporary)
static_assert(
    !std::is_constructible_v<move_only_function<const int&()>, ReturnsInt>,
    "a reference result must not bind to a temporary");
static_assert(
    !std::is_constructible_v<move_only_function<const int&()>, std::in_place_type_t<ReturnsInt>>,
    "a reference result must not bind to a temporary");
#endif

// Counts live instances, moves and copies, so every construction, move and destruction is accounted for.
struct LifetimeCounter {
    static inline int live = 0;
    static inline int moves = 0;
    static inline int copies = 0;
    static void reset() noexcept { live = moves = copies = 0; }
    LifetimeCounter() noexcept { ++live; }
    LifetimeCounter(const LifetimeCounter&) noexcept {
        ++live;
        ++copies;
    }
    LifetimeCounter(LifetimeCounter&&) noexcept {
        ++live;
        ++moves;
    }
    LifetimeCounter& operator=(const LifetimeCounter&) = delete;
    LifetimeCounter& operator=(LifetimeCounter&&) = delete;
    ~LifetimeCounter() { --live; }
};

int answer() { return 42; }

// Construction and invocation.

TEST(MoveOnlyFunctionInvoke, CPU_InvokesLambda) {
    int calls = 0;
    move_only_function<void()> f{[&calls]() { ++calls; }};
    f();
    f();
    EXPECT_EQ(calls, 2);
}

TEST(MoveOnlyFunctionInvoke, CPU_ReturnsValueAndTakesArguments) {
    move_only_function<std::uint64_t(std::uint64_t, std::uint64_t)> add{
        [](std::uint64_t a, std::uint64_t b) { return a + b; }};
    EXPECT_EQ(add(2, 3), 5u);
}

// Only into the target's own parameter, as with std::function: the table passes class types by reference.
TEST(MoveOnlyFunctionInvoke, CPU_ClassArgumentIsMovedOnce) {
    // NOLINTNEXTLINE(performance-unnecessary-value-param): the by-value parameter is what is measured
    move_only_function<void(LifetimeCounter)> f{[](LifetimeCounter) {}};
    LifetimeCounter::reset();
    f(LifetimeCounter{});
    EXPECT_EQ(LifetimeCounter::moves, 1);
    EXPECT_EQ(LifetimeCounter::copies, 0);
}

TEST(MoveOnlyFunctionInvoke, CPU_HoldsMoveOnlyCapture) {
    auto owned = std::make_unique<int>(7);
    move_only_function<int()> f{[owned = std::move(owned)]() { return *owned; }};
    EXPECT_EQ(f(), 7);
}

struct IntHolder {
    int value = 7;
    operator const int&() const { return value; }
};
TEST(MoveOnlyFunctionInvoke, CPU_ReturnsReferenceWithoutTemporary) {
    IntHolder holder;
    move_only_function<const int&()> direct{[&holder]() -> const int& { return holder.value; }};
    EXPECT_EQ(&direct(), &holder.value);
    move_only_function<const int&()> converted{[&holder]() -> IntHolder& { return holder; }};
    EXPECT_EQ(&converted(), &holder.value);
}

// A function name is a function reference, which cannot be null; gcc rejects comparing it to nullptr.
TEST(MoveOnlyFunctionInvoke, CPU_ConstructsFromFunctionName) {
    move_only_function<int()> f{answer};
    EXPECT_EQ(f(), 42);
    move_only_function<int()> g;
    g = answer;
    EXPECT_EQ(g(), 42);
}

TEST(MoveOnlyFunctionInvoke, CPU_NonNullFunctionPointerIsEngaged) {
    move_only_function<int()> f{&answer};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_EQ(f(), 42);
}

TEST(MoveOnlyFunctionInvoke, CPU_NonEmptyStdFunctionIsEngaged) {
    std::function<int()> src{[]() { return 3; }};
    move_only_function<int()> f{src};
    ASSERT_TRUE(static_cast<bool>(f));
    EXPECT_EQ(f(), 3);
}

TEST(MoveOnlyFunctionInvoke, CPU_InPlaceConstructsTarget) {
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

TEST(MoveOnlyFunctionInvoke, CPU_AssigningCallableReplacesTarget) {
    move_only_function<int()> f{[]() { return 1; }};
    f = [p = std::make_unique<int>(2)] { return *p; };
    EXPECT_EQ(f(), 2);
}

// Empty states, and what calling one does.

TEST(MoveOnlyFunctionEmpty, CPU_EmptyByDefault) {
    move_only_function<void()> f;
    EXPECT_FALSE(static_cast<bool>(f));
    EXPECT_TRUE(f == nullptr);
    EXPECT_FALSE(static_cast<bool>(f));
}

TEST(MoveOnlyFunctionEmpty, CPU_EngagedReportsNonEmpty) {
    // Captureless, hence trivially destructible and stored with every fast-path bit set: the case
    // most likely to be mistaken for the empty state.
    move_only_function<void()> f{[]() {}};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_FALSE(f == nullptr);
    EXPECT_TRUE(static_cast<bool>(f));
}

TEST(MoveOnlyFunctionEmpty, CPU_NullFunctionPointerIsEmpty) {
    void (*fp)() = nullptr;
    move_only_function<void()> f{fp};
    EXPECT_FALSE(static_cast<bool>(f));
    EXPECT_TRUE(f == nullptr);
    EXPECT_THROW(f(), std::bad_function_call);
}

TEST(MoveOnlyFunctionEmpty, CPU_NullptrConstructsEmpty) {
    move_only_function<void()> f{nullptr};
    EXPECT_FALSE(static_cast<bool>(f));
}

// As with std::move_only_function, an empty std::function is an ordinary target, not an empty source.
TEST(MoveOnlyFunctionEmpty, CPU_EmptyStdFunctionIsEngaged) {
    std::function<void()> empty;
    move_only_function<void()> f{empty};
    EXPECT_TRUE(static_cast<bool>(f));
    EXPECT_THROW(f(), std::bad_function_call);
}

TEST(MoveOnlyFunctionEmpty, CPU_EmptyOtherSignatureIsEmpty) {
    move_only_function<int()> empty;
    move_only_function<long()> constructed{std::move(empty)};
    EXPECT_FALSE(static_cast<bool>(constructed));

    move_only_function<int()> also_empty;
    move_only_function<long()> assigned{[] { return 1L; }};
    assigned = std::move(also_empty);
    EXPECT_FALSE(static_cast<bool>(assigned));

    move_only_function<int()> engaged{[] { return 5; }};
    move_only_function<long()> converted{std::move(engaged)};
    EXPECT_EQ(converted(), 5);
}

TEST(MoveOnlyFunctionEmpty, CPU_AssigningNullptrEmpties) {
    move_only_function<int()> f{[]() { return 1; }};
    f = nullptr;
    EXPECT_FALSE(static_cast<bool>(f));
    EXPECT_TRUE(nullptr == f);
}

// Pinned deliberately: std::move_only_function makes this undefined, so at the C++23 switch this
// test failing is the intended signal.
TEST(MoveOnlyFunctionEmpty, CPU_CallingEmptyThrows) {
    move_only_function<void()> f;
    EXPECT_THROW(f(), std::bad_function_call);
}

// Moves and swap.

TEST(MoveOnlyFunctionMove, CPU_MoveConstructionTransfersTarget) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst{std::move(src)};
    EXPECT_TRUE(static_cast<bool>(dst));
    EXPECT_EQ(dst(), 42);
}

TEST(MoveOnlyFunctionMove, CPU_MoveAssignmentReplacesTarget) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst{[]() { return 1; }};
    dst = std::move(src);
    EXPECT_EQ(dst(), 42);
}

TEST(MoveOnlyFunctionMove, CPU_MovedFromIsEmpty) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst{std::move(src)};
    EXPECT_FALSE(static_cast<bool>(src));  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(src == nullptr);           // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(static_cast<bool>(dst));
}

TEST(MoveOnlyFunctionMove, CPU_MoveAssignLeavesSourceEmpty) {
    move_only_function<int()> src{[]() { return 42; }};
    move_only_function<int()> dst;
    dst = std::move(src);
    EXPECT_FALSE(static_cast<bool>(src));  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(dst(), 42);
}

TEST(MoveOnlyFunctionMove, CPU_CallingMovedFromThrows) {
    move_only_function<int()> src{[p = std::make_unique<int>(42)] { return *p; }};
    move_only_function<int()> dst{std::move(src)};
    ASSERT_FALSE(static_cast<bool>(src));         // NOLINT(bugprone-use-after-move)
    EXPECT_THROW(src(), std::bad_function_call);  // NOLINT(bugprone-use-after-move)
}

TEST(MoveOnlyFunctionMove, CPU_CallingMoveAssignedFromThrows) {
    move_only_function<int()> src{[p = std::make_unique<int>(42)] { return *p; }};
    move_only_function<int()> dst;
    dst = std::move(src);
    ASSERT_FALSE(static_cast<bool>(src));         // NOLINT(bugprone-use-after-move)
    EXPECT_THROW(src(), std::bad_function_call);  // NOLINT(bugprone-use-after-move)
}

TEST(MoveOnlyFunctionMove, CPU_CallingMovedFromHeapTargetThrows) {
    std::uint64_t big[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    move_only_function<std::uint64_t()> src{[big]() { return big[0]; }};
    move_only_function<std::uint64_t()> dst{std::move(src)};
    ASSERT_FALSE(static_cast<bool>(src));         // NOLINT(bugprone-use-after-move)
    EXPECT_THROW(src(), std::bad_function_call);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(dst(), 1u);
}

// Heap-stored targets corrupt on self-move without the guard in operator=.
TEST(MoveOnlyFunctionMove, CPU_SelfMoveAssignIsSafe) {
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

// The source is owned by the target it replaces, so it must be read before that target is destroyed.
TEST(MoveOnlyFunctionMove, CPU_TargetReplacesItselfWithOwnedSuccessor) {
    move_only_function<int()> heap_outer;
    move_only_function<int()> heap_next{[] { return 2; }};
    heap_outer = [&heap_outer, next = std::move(heap_next)]() mutable {
        heap_outer = std::move(next);
        return 1;
    };
    EXPECT_EQ(heap_outer(), 1);
    EXPECT_EQ(heap_outer(), 2);

    struct OwnsSuccessor {
        move_only_function<int()>* outer;
        std::unique_ptr<move_only_function<int()>> next;
        int operator()() const {
            *outer = std::move(*next);
            return 1;
        }
    };
    static_assert(sizeof(OwnsSuccessor) <= detail::kMoveOnlyFunctionInlineSize);
    static_assert(!std::is_trivially_destructible_v<OwnsSuccessor>);
    move_only_function<int()> inline_outer;
    inline_outer = OwnsSuccessor{&inline_outer, std::make_unique<move_only_function<int()>>([] { return 3; })};
    EXPECT_EQ(inline_outer(), 1);
    EXPECT_EQ(inline_outer(), 3);
}

TEST(MoveOnlyFunctionMove, CPU_SwapExchangesTargets) {
    struct Big {
        std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    };
    move_only_function<std::uint64_t()> inline_target{[]() { return std::uint64_t{3}; }};
    move_only_function<std::uint64_t()> empty;
    inline_target.swap(empty);
    EXPECT_FALSE(static_cast<bool>(inline_target));
    EXPECT_EQ(empty(), 3u);
    swap(inline_target, empty);
    EXPECT_EQ(inline_target(), 3u);
    EXPECT_FALSE(static_cast<bool>(empty));

    move_only_function<std::uint64_t()> heap{[big = Big{}]() { return big.data[7]; }};
    heap.swap(inline_target);
    EXPECT_EQ(heap(), 3u);
    EXPECT_EQ(inline_target(), 8u);
    swap(heap, inline_target);
    EXPECT_EQ(heap(), 8u);
    EXPECT_EQ(inline_target(), 3u);

    move_only_function<std::uint64_t()> other_heap{[big = Big{}]() { return big.data[0]; }};
    heap.swap(other_heap);
    EXPECT_EQ(heap(), 1u);
    EXPECT_EQ(other_heap(), 8u);
}

// Where a target lives: the inline buffer or the heap.

TEST(MoveOnlyFunctionStorage, CPU_HoldsCaptureLargerThanInlineBuffer) {
    struct Big {
        std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    };
    static_assert(sizeof(Big) > detail::kMoveOnlyFunctionInlineSize);
    Big big{};
    move_only_function<std::uint64_t()> f{[big]() { return big.data[7]; }};
    EXPECT_EQ(f(), 8u);
}

TEST(MoveOnlyFunctionStorage, CPU_ThrowingMoveTargetStaysOnHeap) {
    struct Target {
        Target() = default;
        Target(Target&&) { throw 1; }  // NOLINT(performance-noexcept-move-constructor): the point of the test
        int operator()() const { return 42; }
    };
    static_assert(sizeof(Target) <= detail::kMoveOnlyFunctionInlineSize);
    static_assert(alignof(Target) <= detail::kMoveOnlyFunctionInlineAlign);
    static_assert(!std::is_nothrow_move_constructible_v<Target>);

    move_only_function<int()> src{std::in_place_type<Target>};
    move_only_function<int()> dst{std::move(src)};
    EXPECT_EQ(dst(), 42);
    move_only_function<int()> assigned{[] { return 0; }};
    assigned = std::move(dst);
    EXPECT_EQ(assigned(), 42);
}

// Too aligned for the buffer, so it goes to the heap: the allocation honours the alignment, and a move
// keeps the target where it is.
TEST(MoveOnlyFunctionStorage, CPU_OverAlignedTargetGoesToHeap) {
    struct alignas(16) OverAligned {
        std::uintptr_t operator()() const { return reinterpret_cast<std::uintptr_t>(this); }
    };
    // On 64-bit it fits the buffer, so its alignment alone sends it to the heap.
    static_assert(sizeof(void*) < 8 || sizeof(OverAligned) <= detail::kMoveOnlyFunctionInlineSize);
    static_assert(alignof(OverAligned) > detail::kMoveOnlyFunctionInlineAlign);
    move_only_function<std::uintptr_t()> constructed{OverAligned{}};
    const std::uintptr_t address = constructed();
    EXPECT_EQ(address % alignof(OverAligned), 0u);
    move_only_function<std::uintptr_t()> moved{std::move(constructed)};
    EXPECT_EQ(moved(), address);
}

// Target lifetimes: each target is moved by its own constructor and destroyed exactly once.

// A target that points at itself sees a byte copy as a stale pointer, which counting constructions
// and destructions cannot.
struct SelfReferential {
    SelfReferential() noexcept : self(this) {}
    SelfReferential(SelfReferential&&) noexcept : self(this) {}
    SelfReferential& operator=(SelfReferential&&) = delete;
    bool operator()() const { return self == this; }
    const SelfReferential* self;
};
TEST(MoveOnlyFunctionLifetime, CPU_NonTrivialInlineTargetIsMovedByItsConstructor) {
    static_assert(sizeof(SelfReferential) <= detail::kMoveOnlyFunctionInlineSize);
    static_assert(!std::is_trivially_move_constructible_v<SelfReferential>);
    move_only_function<bool()> constructed{std::in_place_type<SelfReferential>};
    EXPECT_TRUE(constructed());
    move_only_function<bool()> moved{std::move(constructed)};
    EXPECT_TRUE(moved());
    move_only_function<bool()> assigned{[] { return false; }};
    assigned = std::move(moved);
    EXPECT_TRUE(assigned());
}

// Inline but not trivially movable: moves go through the per-type relocate, and each moved-out
// copy must be destroyed exactly once.
TEST(MoveOnlyFunctionLifetime, CPU_NonTrivialInlineTargetIsDestroyedOnce) {
    LifetimeCounter::reset();
    {
        move_only_function<int()> a{[c = LifetimeCounter{}]() { return 1; }};
        EXPECT_EQ(LifetimeCounter::live, 1);
        move_only_function<int()> b{std::move(a)};
        EXPECT_EQ(LifetimeCounter::live, 1);
        move_only_function<int()> c{[]() { return 2; }};
        c = std::move(b);
        EXPECT_EQ(LifetimeCounter::live, 1);
        EXPECT_EQ(c(), 1);
    }
    EXPECT_EQ(LifetimeCounter::live, 0);
}

// Heap targets move by pointer: the target is neither copied nor destroyed by a move, and the one
// being replaced is destroyed.
TEST(MoveOnlyFunctionLifetime, CPU_HeapTargetMovesWithoutCopying) {
    struct Big {
        LifetimeCounter counter;
        std::uint64_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    };
    static_assert(sizeof(Big) > detail::kMoveOnlyFunctionInlineSize);
    LifetimeCounter::reset();
    {
        move_only_function<std::uint64_t()> a{[big = Big{}]() { return big.data[7]; }};
        move_only_function<std::uint64_t()> b{[big = Big{}]() { return big.data[0]; }};
        EXPECT_EQ(LifetimeCounter::live, 2);
        b = std::move(a);
        EXPECT_EQ(LifetimeCounter::live, 1);
        move_only_function<std::uint64_t()> c{std::move(b)};
        EXPECT_EQ(LifetimeCounter::live, 1);
        EXPECT_EQ(c(), 8u);
    }
    EXPECT_EQ(LifetimeCounter::live, 0);
}

}  // namespace
}  // namespace ttsl
