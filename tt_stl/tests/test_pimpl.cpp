// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/pimpl.hpp>

#include <type_traits>
#include <utility>

#include "test_indirect_util/widget.hpp"

namespace ttsl {
namespace {

// Widget/MoveOnlyWidget derive from PimplBase<WidgetImpl>, where WidgetImpl is only
// forward-declared in widget.hpp and defined in widget.cpp -- proving PimplBase (not just
// indirect) preserves the incomplete-type/linker firewall.

TEST(PimplBaseTest, DeepCopy) {
    Widget w1(42);
    Widget w2(w1);

    EXPECT_EQ(w1.value(), 42);
    EXPECT_EQ(w2.value(), 42);

    // Value semantics: modifying one doesn't affect the other.
    w1.set_value(100);
    EXPECT_EQ(w1.value(), 100);
    EXPECT_EQ(w2.value(), 42);
}

TEST(PimplBaseTest, CopyAssignmentDeepCopies) {
    Widget w1(42);
    Widget w2(100);

    w2 = w1;
    EXPECT_EQ(w2.value(), 42);

    w1.set_value(200);
    EXPECT_EQ(w1.value(), 200);
    EXPECT_EQ(w2.value(), 42);
}

TEST(PimplBaseTest, MoveLeavesSourceValueless) {
    Widget w1(42);
    Widget w2(std::move(w1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(w1.valueless_after_move());
    EXPECT_FALSE(w2.valueless_after_move());
    EXPECT_EQ(w2.value(), 42);
}

TEST(PimplBaseTest, MoveAssignmentLeavesSourceValueless) {
    Widget w1(42);
    Widget w2(100);

    w2 = std::move(w1);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(w1.valueless_after_move());
    EXPECT_EQ(w2.value(), 42);
}

TEST(PimplBaseTest, CheckedImplAccessOnValuelessFatals) {
    Widget w1(42);
    Widget w2(std::move(w1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_ANY_THROW(w1.value());
}

TEST(PimplBaseTest, MoveOnlyWidgetCompilesAndMoves) {
    static_assert(!std::is_copy_constructible_v<MoveOnlyWidget>);
    static_assert(std::is_move_constructible_v<MoveOnlyWidget>);

    MoveOnlyWidget w1(7);
    MoveOnlyWidget w2(std::move(w1));

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_TRUE(w1.valueless_after_move());
    EXPECT_EQ(w2.value(), 7);
}

}  // namespace
}  // namespace ttsl
