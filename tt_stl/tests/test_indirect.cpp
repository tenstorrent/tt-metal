// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/pimpl.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace ttsl {
namespace {

// Helper class for testing.
struct TestValue {
    int value{};

    TestValue() = default;
    explicit TestValue(int v) : value(v) {}
};

TEST(IndirectTest, InPlaceConstruction) {
    indirect<std::string> s(std::in_place, "hello");
    EXPECT_FALSE(s.valueless_after_move());
    EXPECT_EQ(*s, "hello");
}

TEST(IndirectTest, InPlaceConstructionMultipleArgs) {
    indirect<std::string> s(std::in_place, 5, 'x');
    EXPECT_FALSE(s.valueless_after_move());
    EXPECT_EQ(*s, "xxxxx");
}

TEST(IndirectTest, CopyConstruction) {
    indirect<TestValue> i1(std::in_place, 42);
    indirect<TestValue> i2(i1);

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i1->value, 42);
    EXPECT_EQ(i2->value, 42);

    // Value semantics: modifying one doesn't affect the other.
    i1->value = 100;
    EXPECT_EQ(i1->value, 100);
    EXPECT_EQ(i2->value, 42);
}

TEST(IndirectTest, CopyConstructionFromValueless) {
    indirect<int> i1(std::in_place, 42);
    indirect<int> i2(std::move(i1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    // Copying a valueless indirect produces another valueless indirect.
    indirect<int> i3(i1);
    EXPECT_TRUE(i3.valueless_after_move());
}

TEST(IndirectTest, MoveConstruction) {
    indirect<TestValue> i1(std::in_place, 42);
    indirect<TestValue> i2(std::move(i1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i2->value, 42);
}

TEST(IndirectTest, CopyAssignment) {
    indirect<TestValue> i1(std::in_place, 42);
    indirect<TestValue> i2(std::in_place, 100);

    i2 = i1;

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i1->value, 42);
    EXPECT_EQ(i2->value, 42);
}

TEST(IndirectTest, CopyAssignmentFromValueless) {
    indirect<int> i1(std::in_place, 42);
    indirect<int> i2(std::move(i1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    indirect<int> i3(std::in_place, 100);
    i3 = i1;

    EXPECT_TRUE(i3.valueless_after_move());
}

TEST(IndirectTest, CopyAssignmentToValueless) {
    indirect<int> i1(std::in_place, 42);
    indirect<int> i2(std::move(i1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    indirect<int> i3(std::in_place, 100);
    i1 = i3;

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_EQ(*i1, 100);
}

TEST(IndirectTest, CopyAssignmentStrongExceptionGuarantee) {
    // Copy-constructing from an instance with throw_on_copy set simulates a T whose copy
    // constructor can fail (e.g. due to an allocation failure inside T itself).
    struct ThrowingCopy {
        int value{};
        bool throw_on_copy{};

        ThrowingCopy(int v, bool throw_on_copy_) : value(v), throw_on_copy(throw_on_copy_) {}
        ThrowingCopy(const ThrowingCopy& other) : value(other.value), throw_on_copy(other.throw_on_copy) {
            if (other.throw_on_copy) {
                throw std::runtime_error("ThrowingCopy: simulated copy failure");
            }
        }
    };

    indirect<ThrowingCopy> i1(std::in_place, 1, /*throw_on_copy=*/true);
    indirect<ThrowingCopy> i2(std::in_place, 2, /*throw_on_copy=*/false);

    EXPECT_THROW(i2 = i1, std::runtime_error);

    // Strong guarantee: a failed copy-assignment leaves the target untouched (the new T is
    // allocated before p is replaced).
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i2->value, 2);
}

TEST(IndirectTest, SelfCopyAssignment) {
    indirect<int> i(std::in_place, 42);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
    i = i;
#pragma clang diagnostic pop
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 42);
}

TEST(IndirectTest, MoveAssignment) {
    indirect<TestValue> i1(std::in_place, 42);
    indirect<TestValue> i2(std::in_place, 100);

    i2 = std::move(i1);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i2->value, 42);
}

TEST(IndirectTest, MoveAssignmentFromValueless) {
    indirect<int> i1(std::in_place, 42);
    indirect<int> i2(std::move(i1));  // i1 is now valueless.

    indirect<int> i3(std::in_place, 100);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    i3 = std::move(i1);

    EXPECT_TRUE(i3.valueless_after_move());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());
}

TEST(IndirectTest, MoveAssignmentToValueless) {
    indirect<int> i1(std::in_place, 42);
    indirect<int> i2(std::move(i1));  // i1 is now valueless.

    indirect<int> i3(std::in_place, 100);
    i1 = std::move(i3);

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_EQ(*i1, 100);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i3.valueless_after_move());
}

TEST(IndirectTest, SelfMoveAssignment) {
    indirect<int> i(std::in_place, 42);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    i = std::move(i);
#pragma clang diagnostic pop
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 42);
}

TEST(IndirectTest, DereferenceOperators) {
    indirect<int> i(std::in_place, 42);

    // Non-const lvalue reference.
    EXPECT_EQ(*i, 42);
    *i = 100;
    EXPECT_EQ(*i, 100);

    // Const lvalue reference.
    const indirect<int>& ci = i;
    EXPECT_EQ(*ci, 100);
}

TEST(IndirectTest, ArrowOperator) {
    struct Point {
        int x, y;
        int sum() const { return x + y; }
    };

    indirect<Point> p(std::in_place, Point{3, 4});
    EXPECT_EQ(p->x, 3);
    EXPECT_EQ(p->y, 4);
    EXPECT_EQ(p->sum(), 7);

    p->x = 10;
    EXPECT_EQ(p->x, 10);
}

// Templated on the indirect (or reference-to-indirect) type so the requires-clause below is
// checked via genuine template substitution -- a plain `requires { cm->modify(); }` outside of a
// template is not a SFINAE-friendly context and would hard-error instead of evaluating to false.
template <typename Ind>
concept CanCallModifyThroughArrow = requires(Ind& i) { i->modify(); };

TEST(IndirectTest, ConstPropagation) {
    struct Mutable {
        int value;
        void modify() { value = 100; }
        int get() const { return value; }
    };

    indirect<Mutable> m(std::in_place, Mutable{42});
    EXPECT_EQ(m->get(), 42);

    m->modify();
    EXPECT_EQ(m->get(), 100);

    const indirect<Mutable>& cm = m;
    EXPECT_EQ(cm->get(), 100);

    // Const propagation: operator-> on a const indirect returns a const Mutable*, so calling a
    // non-const member function through it should not compile.
    static_assert(CanCallModifyThroughArrow<indirect<Mutable>>);
    static_assert(!CanCallModifyThroughArrow<const indirect<Mutable>>);
}

TEST(IndirectTest, CheckedDerefOnValuelessFatals) {
    indirect<int> i1(std::in_place, 42);
    indirect<int> i2(std::move(i1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());
    // Unlike std::indirect (UB on valueless), our checked observers TT_FATAL instead.
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_ANY_THROW((void)*i1);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_ANY_THROW((void)i1.operator->());
}

}  // namespace
}  // namespace ttsl
