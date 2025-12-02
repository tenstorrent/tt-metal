// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/indirect.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ttsl {
namespace {

// Helper class for testing
struct TestValue {
    int value{};

    TestValue() = default;
    explicit TestValue(int v) : value(v) {}

    auto operator<=>(const TestValue& other) const = default;
};

// Test default construction
TEST(IndirectTest, DefaultConstruction) {
    indirect<int> i;
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 0);
}

// Test construction with value
TEST(IndirectTest, ConstructionWithValue) {
    indirect<int> i(42);
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 42);
}

// Test in-place construction
TEST(IndirectTest, InPlaceConstruction) {
    indirect<std::string> s(std::in_place, "hello");
    EXPECT_FALSE(s.valueless_after_move());
    EXPECT_EQ(*s, "hello");
}

// Test in-place construction with multiple arguments
TEST(IndirectTest, InPlaceConstructionMultipleArgs) {
    indirect<std::string> s(std::in_place, 5, 'x');
    EXPECT_FALSE(s.valueless_after_move());
    EXPECT_EQ(*s, "xxxxx");
}

// Test in-place construction with initializer list
TEST(IndirectTest, InPlaceConstructionInitializerList) {
    indirect<std::vector<int>> v(std::in_place, {1, 2, 3, 4, 5});
    EXPECT_FALSE(v.valueless_after_move());
    EXPECT_EQ(v->size(), 5);
    EXPECT_EQ(v, std::vector<int>({1, 2, 3, 4, 5}));
}

// Test copy construction
TEST(IndirectTest, CopyConstruction) {
    indirect<TestValue> i1(TestValue(42));
    indirect<TestValue> i2(i1);

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i1->value, 42);
    EXPECT_EQ(i2->value, 42);
    EXPECT_EQ(i1, i2);

    // Test value semantics - modifying one doesn't affect the other
    i1->value = 100;
    EXPECT_EQ(i1->value, 100);
    EXPECT_EQ(i2->value, 42);  // i2 should not be affected
}

// Test copy construction from valueless indirect
TEST(IndirectTest, CopyConstructionFromValueless) {
    indirect<int> i1(42);
    indirect<int> i2(std::move(i1));

    // i1 is now valueless
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    // Copy from valueless should create a valueless indirect
    indirect<int> i3(i1);
    EXPECT_TRUE(i3.valueless_after_move());
}

// Test move construction
TEST(IndirectTest, MoveConstruction) {
    indirect<TestValue> i1(TestValue(42));
    indirect<TestValue> i2(std::move(i1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i2->value, 42);
}

// Test copy assignment
TEST(IndirectTest, CopyAssignment) {
    indirect<TestValue> i1(TestValue(42));
    indirect<TestValue> i2(TestValue(100));

    i2 = i1;

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i1->value, 42);
    EXPECT_EQ(i2->value, 42);
    EXPECT_EQ(i1, i2);
}

// Test copy assignment from valueless to valid
TEST(IndirectTest, CopyAssignmentFromValueless) {
    indirect<int> i1(42);
    indirect<int> i2(std::move(i1));

    // i1 is now valueless
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    indirect<int> i3(100);
    i3 = i1;

    EXPECT_TRUE(i3.valueless_after_move());
}

// Test copy assignment from valid to valueless
TEST(IndirectTest, CopyAssignmentToValueless) {
    indirect<int> i1(42);
    indirect<int> i2(std::move(i1));

    // i1 is now valueless
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    indirect<int> i3(100);
    i1 = i3;

    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_EQ(*i1, 100);
}

// Test self-assignment
TEST(IndirectTest, SelfCopyAssignment) {
    indirect<int> i(42);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
    i = i;
#pragma clang diagnostic pop
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 42);
}

// Test move assignment
TEST(IndirectTest, MoveAssignment) {
    indirect<TestValue> i1(TestValue(42));
    indirect<TestValue> i2(TestValue(100));

    i2 = std::move(i1);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());
    EXPECT_EQ(i2->value, 42);
}

// Test self-move-assignment
TEST(IndirectTest, SelfMoveAssignment) {
    indirect<int> i(42);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    i = std::move(i);
#pragma clang diagnostic pop
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 42);
}

// Test value assignment
TEST(IndirectTest, ValueAssignment) {
    indirect<int> i(42);
    i = 100;
    EXPECT_FALSE(i.valueless_after_move());
    EXPECT_EQ(*i, 100);
}

// Test value assignment to valueless indirect
TEST(IndirectTest, ValueAssignmentToValueless) {
    indirect<int> i1(42);
    indirect<int> i2(std::move(i1));

    // i1 is now valueless
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(i1.valueless_after_move());

    i1 = 100;
    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_EQ(*i1, 100);
}

// Test dereference operators
TEST(IndirectTest, DereferenceOperators) {
    indirect<int> i(42);

    // Non-const lvalue reference
    EXPECT_EQ(*i, 42);
    *i = 100;
    EXPECT_EQ(*i, 100);

    // Const lvalue reference
    const indirect<int>& ci = i;
    EXPECT_EQ(*ci, 100);

    // Rvalue reference
    indirect<int> i2(42);
    int&& rv = *std::move(i2);
    EXPECT_EQ(rv, 42);
}

// Test arrow operator
TEST(IndirectTest, ArrowOperator) {
    struct Point {
        int x, y;
        int sum() const { return x + y; }
    };

    indirect<Point> p(Point{3, 4});
    EXPECT_EQ(p->x, 3);
    EXPECT_EQ(p->y, 4);
    EXPECT_EQ(p->sum(), 7);

    p->x = 10;
    EXPECT_EQ(p->x, 10);
}

// Test const propagation
TEST(IndirectTest, ConstPropagation) {
    struct Mutable {
        int value;
        void modify() { value = 100; }
        int get() const { return value; }
    };

    indirect<Mutable> m(Mutable{42});
    EXPECT_EQ(m->get(), 42);

    m->modify();
    EXPECT_EQ(m->get(), 100);

    const indirect<Mutable>& cm = m;
    EXPECT_EQ(cm->get(), 100);
    // The following should not compile (const propagation):
    // cm->modify();
}

// Test swap member function
TEST(IndirectTest, SwapMember) {
    indirect<int> i1(42);
    indirect<int> i2(100);

    i1.swap(i2);

    EXPECT_EQ(*i1, 100);
    EXPECT_EQ(*i2, 42);
}

// Test swap free function
TEST(IndirectTest, SwapFree) {
    indirect<int> i1(42);
    indirect<int> i2(100);

    swap(i1, i2);

    EXPECT_EQ(*i1, 100);
    EXPECT_EQ(*i2, 42);
}

// Test swap with valueless
TEST(IndirectTest, SwapWithValueless) {
    indirect<int> i1(42);
    indirect<int> temp(100);
    indirect<int> i2(std::move(temp));

    // i2's source (temp) is valueless
    EXPECT_FALSE(i1.valueless_after_move());
    EXPECT_FALSE(i2.valueless_after_move());

    swap(i1, i2);

    EXPECT_EQ(*i1, 100);
    EXPECT_EQ(*i2, 42);
}

// Test equality operator
TEST(IndirectTest, EqualityOperator) {
    indirect<int> i1(42);
    indirect<int> i2(42);
    indirect<int> i3(100);

    EXPECT_EQ(i1, i2);
    EXPECT_NE(i1, i3);
}

// Test comparison with different types
TEST(IndirectTest, ComparisonDifferentTypes) {
    indirect<int> i1(42);
    indirect<double> d1(42.0);
    indirect<double> d2(100.0);

    EXPECT_EQ(i1, d1);
    EXPECT_NE(i1, d2);
}

// Test spaceship operator
TEST(IndirectTest, SpaceshipOperator) {
    indirect<int> i1(42);
    indirect<int> i2(100);

    EXPECT_LT(i1, i2);
    EXPECT_GT(i2, i1);
    EXPECT_LE(i1, i2);
    EXPECT_GE(i2, i1);
}

// Test comparison with T directly
TEST(IndirectTest, ComparisonWithT) {
    indirect<int> i(42);

    EXPECT_EQ(i, 42);
    EXPECT_NE(i, 100);
    EXPECT_LT(i, 100);
    EXPECT_GT(i, 10);
}

// Test deduction guide
TEST(IndirectTest, DeductionGuide) {
    auto i = indirect(42);
    static_assert(requires {
        { indirect(42) } -> std::same_as<indirect<int>>;
    });
    EXPECT_EQ(*i, 42);

    auto s = indirect(std::string("hello"));
    static_assert(std::is_same_v<decltype(s), indirect<std::string>>);
    EXPECT_EQ(*s, "hello");
}

// TODO: Add test for incomplete type support (PImpl pattern)
// This would require a more complex test setup with separate compilation units

// Test with polymorphic types
TEST(IndirectTest, PolymorphicTypes) {
    struct Base {
        int value;
        explicit Base(int v) : value(v) {}

        Base(const Base&) = default;
        Base(Base&&) noexcept = default;

        Base& operator=(const Base&) = default;
        Base& operator=(Base&&) noexcept = default;

        virtual ~Base() = default;
        virtual int get() const { return value; }
    };

    struct Derived : Base {
        explicit Derived(int v) : Base(v) {}
        int get() const override { return value * 2; }
    };

    indirect<Base> b(Base{42});
    EXPECT_EQ(b->get(), 42);

    indirect<Derived> d(Derived{21});
    EXPECT_EQ(d->get(), 42);
}

// Test move semantics with rvalue dereference
TEST(IndirectTest, RvalueDereference) {
    struct MoveOnly {
        std::unique_ptr<int> ptr;
        explicit MoveOnly(int v) : ptr(std::make_unique<int>(v)) {}
    };

    indirect<MoveOnly> i(MoveOnly{42});
    MoveOnly extracted = std::move(i);

    EXPECT_EQ(*extracted.ptr, 42);
}

}  // namespace
}  // namespace ttsl
