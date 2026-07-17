// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt_stl/indestructible.hpp>
#include <memory>

namespace ttsl {
namespace {

TEST(IndestructibleTest, Basic) {
    struct DangerouslyDestructible {
        ~DangerouslyDestructible() {
            // Wrapping in a lambda, as `FAIL()` returns `void`.
            []() { FAIL(); }();
        }
    };

    Indestructible<DangerouslyDestructible> obj;
}

}  // namespace
}  // namespace ttsl
