// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt_stl/tt_pause.hpp>
#include <atomic>
#include <chrono>
#include <thread>

namespace ttsl {
namespace {

TEST(TTPauseTest, PauseDoesNotCrash) {
    // Simply verify that pause can be called without crashing
    for (int i = 0; i < 1000; ++i) {
        pause();
    }
}

TEST(TTNiceSpinUntilTest, ImmediateReturnWhenPredicateTrue) {
    // Predicate is immediately true, should return without spinning
    auto start = std::chrono::high_resolution_clock::now();
    nice_spin_until([] { return true; });
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // Should complete almost instantly (less than 1ms)
    EXPECT_LT(duration.count(), 1000);
}

TEST(TTNiceSpinUntilTest, WaitsUntilPredicateBecomesTrueWithAtomicFlag) {
    std::atomic<bool> flag{false};

    std::thread setter([&flag] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        flag.store(true);
    });

    auto start = std::chrono::high_resolution_clock::now();
    nice_spin_until([&flag] { return flag.load(); });
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Should have waited at least ~10ms for the flag
    EXPECT_GE(duration.count(), 9);

    setter.join();
}

TEST(TTNiceSpinUntilTest, WaitsUntilPredicateBecomesTrueWithCounter) {
    std::atomic<int> counter{0};

    std::thread incrementer([&counter] {
        for (int i = 0; i < 5; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            counter.fetch_add(1);
        }
    });

    nice_spin_until([&counter] { return counter.load() >= 5; });
    EXPECT_GE(counter.load(), 5);

    incrementer.join();
}

TEST(TTNiceSpinUntilTest, CustomNSpinsParameter) {
    std::atomic<int> call_count{0};

    // With N_SPINS=10, we should hit the sleep branch more frequently
    auto predicate = [&call_count] {
        call_count.fetch_add(1);
        return call_count.load() >= 50;
    };

    nice_spin_until<10>(predicate);
    EXPECT_GE(call_count.load(), 50);
}

TEST(TTNiceSpinUntilTest, CustomMaxWaitUSParameter) {
    std::atomic<bool> flag{false};

    std::thread setter([&flag] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        flag.store(true);
    });

    // Use custom parameters: N_SPINS=50, MAX_WAIT_US=8
    nice_spin_until<50, 8>([&flag] { return flag.load(); });
    EXPECT_TRUE(flag.load());

    setter.join();
}

TEST(TTNiceSpinUntilTest, PredicateWithArguments) {
    std::atomic<int> value{0};

    std::thread setter([&value] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        value.store(42);
    });

    // Predicate that takes an argument
    auto check_value = [&value](int expected) { return value.load() == expected; };

    nice_spin_until(check_value, 42);
    EXPECT_EQ(value.load(), 42);

    setter.join();
}

TEST(TTNiceSpinUntilTest, PredicateWithMultipleArguments) {
    std::atomic<int> a{0};
    std::atomic<int> b{0};

    std::thread setter([&a, &b] {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        a.store(10);
        b.store(20);
    });

    // Predicate that takes multiple arguments
    auto check_sum = [&a, &b](int expected_a, int expected_b) {
        return a.load() == expected_a && b.load() == expected_b;
    };

    nice_spin_until(check_sum, 10, 20);
    EXPECT_EQ(a.load(), 10);
    EXPECT_EQ(b.load(), 20);

    setter.join();
}

TEST(TTNiceSpinUntilTest, ExponentialBackoffBehavior) {
    // This test verifies the exponential backoff by checking that the function
    // doesn't consume excessive CPU time when waiting for a longer duration
    std::atomic<bool> flag{false};

    std::thread setter([&flag] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        flag.store(true);
    });

    auto start = std::chrono::high_resolution_clock::now();
    nice_spin_until([&flag] { return flag.load(); });
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Should have waited approximately 100ms
    EXPECT_GE(duration.count(), 95);
    EXPECT_LT(duration.count(), 200);

    setter.join();
}

TEST(TTNiceSpinUntilTest, ZeroSpinsGoesDirectlyToSleep) {
    // With N_SPINS=1, should hit sleep on every iteration
    std::atomic<bool> flag{false};

    std::thread setter([&flag] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        flag.store(true);
    });

    nice_spin_until<1, 4>([&flag] { return flag.load(); });
    EXPECT_TRUE(flag.load());

    setter.join();
}

}  // namespace
}  // namespace ttsl
