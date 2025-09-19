// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include <tt_stl/cleanup.hpp>
#include <memory>
#include <vector>

namespace ttsl {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(CleanupTest, Basic) {
    int counter = 0;
    {
        auto cleanup = make_cleanup([&counter]() { counter++; });
        EXPECT_EQ(counter, 0);
    }
    EXPECT_EQ(counter, 1);
}

TEST(CleanupTest, MultipleCleanupsInOrder) {
    std::vector<int> log;
    {
        auto cleanup1 = make_cleanup([&log]() { log.push_back(1); });
        auto cleanup2 = make_cleanup([&log]() { log.push_back(2); });
        auto cleanup3 = make_cleanup([&log]() { log.push_back(3); });
    }
    EXPECT_THAT(log, ElementsAre(3, 2, 1));
}

TEST(CleanupTest, CancelCleanup) {
    int counter = 0;
    {
        auto cleanup = make_cleanup([&counter]() { counter++; });
        EXPECT_EQ(counter, 0);
        std::move(cleanup).cancel();
    }
    EXPECT_EQ(counter, 0);
}

TEST(CleanupTest, MoveConstruction) {
    int counter = 0;
    {
        auto cleanup1 = make_cleanup([&counter]() { counter++; });
        auto cleanup2 = std::move(cleanup1);
        EXPECT_EQ(counter, 0);
    }
    EXPECT_EQ(counter, 1);
}

TEST(CleanupTest, ExceptionSafety) {
    bool cleanup_called = false;
    try {
        auto cleanup = make_cleanup([&cleanup_called]() { cleanup_called = true; });
        throw std::runtime_error("test exception");
    } catch (const std::runtime_error& e) {
        // Exception caught, cleanup should still execute
        EXPECT_THAT(e.what(), HasSubstr("test exception"));
    }
    EXPECT_TRUE(cleanup_called);
}

TEST(CleanupTest, PerfectForwarding) {
    std::vector<int> captured = {1, 2, 3};
    {
        auto cleanup = make_cleanup([captured = std::move(captured)]() {
            // captured should be moved into the lambda
        });
        // NOLINTNEXTLINE(bugprone-use-after-move)
        EXPECT_TRUE(captured.empty());  // Should be moved from
    }
}

}  // namespace
}  // namespace ttsl
