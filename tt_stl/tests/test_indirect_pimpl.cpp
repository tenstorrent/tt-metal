// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/indirect.hpp>

#include "test_indirect_util/widget.hpp"

namespace ttsl {
namespace {

// These tests demonstrate that indirect<Widget> works with incomplete types!
// Widget is only forward-declared in widget.hpp - the full definition
// is in widget.cpp which is compiled separately and linked.
// This proves that indirect can hold and manage incomplete types.

// Test basic construction with incomplete type
TEST(IndirectPImplTest, BasicConstruction) {
    // Widget is incomplete here, but indirect<Widget> works!
    indirect<Widget> w = createWidget();
    EXPECT_EQ(getWidgetValue(w), 0);
}

// Test construction with value
TEST(IndirectPImplTest, ConstructionWithValue) {
    indirect<Widget> w = createWidget(42);
    EXPECT_EQ(getWidgetValue(w), 42);
}

// Test copy construction
TEST(IndirectPImplTest, CopyConstruction) {
    indirect<Widget> w1 = createWidget(42);
    indirect<Widget> w2 = w1;

    EXPECT_EQ(getWidgetValue(w1), 42);
    EXPECT_EQ(getWidgetValue(w2), 42);

    // Verify deep copy - modifying one doesn't affect the other
    setWidgetValue(w1, 100);
    EXPECT_EQ(getWidgetValue(w1), 100);
    EXPECT_EQ(getWidgetValue(w2), 42);
}

// Test move construction
TEST(IndirectPImplTest, MoveConstruction) {
    indirect<Widget> w1 = createWidget(42);
    indirect<Widget> w2 = std::move(w1);

    EXPECT_TRUE(w1.valueless_after_move());
    EXPECT_EQ(getWidgetValue(w2), 42);
}

// Test copy assignment
TEST(IndirectPImplTest, CopyAssignment) {
    indirect<Widget> w1 = createWidget(42);
    indirect<Widget> w2 = createWidget(100);

    w2 = w1;

    EXPECT_EQ(getWidgetValue(w1), 42);
    EXPECT_EQ(getWidgetValue(w2), 42);

    // Verify deep copy
    setWidgetValue(w1, 200);
    EXPECT_EQ(getWidgetValue(w1), 200);
    EXPECT_EQ(getWidgetValue(w2), 42);
}

// Test move assignment
TEST(IndirectPImplTest, MoveAssignment) {
    indirect<Widget> w1 = createWidget(42);
    indirect<Widget> w2 = createWidget(100);

    w2 = std::move(w1);

    EXPECT_TRUE(w1.valueless_after_move());
    EXPECT_EQ(getWidgetValue(w2), 42);
}

// Test dereference and manipulation
TEST(IndirectPImplTest, DereferenceAndManipulate) {
    indirect<Widget> w = createWidget(123);

    EXPECT_EQ(getWidgetValue(w), 123);

    setWidgetValue(w, 456);
    EXPECT_EQ(getWidgetValue(w), 456);
}

// Test const correctness
TEST(IndirectPImplTest, ConstCorrectness) {
    indirect<Widget> w = createWidget(42);
    const indirect<Widget>& cw = w;

    EXPECT_EQ(getWidgetValue(cw), 42);

    setWidgetValue(w, 100);
    EXPECT_EQ(getWidgetValue(cw), 100);
}

// Test multiple operations
TEST(IndirectPImplTest, MultipleOperations) {
    indirect<Widget> w1 = createWidget(10);
    indirect<Widget> w2 = createWidget(20);
    indirect<Widget> w3 = w1;

    setWidgetValue(w1, getWidgetValue(w1) + getWidgetValue(w2));
    EXPECT_EQ(getWidgetValue(w1), 30);
    EXPECT_EQ(getWidgetValue(w2), 20);
    EXPECT_EQ(getWidgetValue(w3), 10);

    w3 = w2;
    EXPECT_EQ(getWidgetValue(w3), 20);
}

}  // namespace
}  // namespace ttsl
