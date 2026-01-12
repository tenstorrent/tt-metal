#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>

/**
 * MonotonicQueue: A sliding window maximum data structure
 *
 * Maintains the maximum value over a fixed-size sliding window in O(1) amortized time.
 * Uses a monotonic decreasing deque to efficiently track the maximum element.
 *
 * Template parameters:
 *   T: The value type (must support comparison operators)
 *
 * Time complexity:
 *   - push(): O(1) amortized
 *   - max(): O(1)
 *
 * Space complexity: O(k) where k is the window size
 *
 * Example usage:
 *   MonotonicQueue<double> mq(10);  // Window size of 10
 *   mq.push(5.0);
 *   mq.push(3.0);
 *   mq.push(7.0);
 *   double max_val = mq.max();  // Returns 7.0
 */
template <typename T>
class MonotonicQueue {
private:
    struct Element {
        uint64_t index;
        T value;
    };

    std::deque<Element> deque_;
    uint64_t counter_;
    uint64_t window_size_;

public:
    explicit MonotonicQueue(size_t window_size) : counter_(0), window_size_(window_size) {
        if (window_size == 0) {
            throw std::invalid_argument("MonotonicQueue window_size must be > 0");
        }
    }

    /**
     * Pushes a new value into the sliding window
     * Automatically maintains the monotonic decreasing property and window size
     *
     * Note: Overflow-safe - uses unsigned arithmetic where (counter_ - old_index)
     * correctly computes relative distance even when counter_ wraps around.
     */
    void push(const T& value) {
        // Remove smaller elements from back (they can never be maximum)
        while (!deque_.empty() && deque_.back().value <= value) {
            deque_.pop_back();
        }

        // Add current element
        deque_.push_back({counter_++, value});

        // Remove elements outside the window from front
        // Unsigned subtraction handles counter overflow correctly via modular arithmetic
        while (!deque_.empty() && (counter_ - deque_.front().index) > window_size_) {
            deque_.pop_front();
        }
    }

    /**
     * Returns the maximum value in the current window
     * Precondition: The queue must not be empty (call empty() to check)
     */
    const T& max() const {
        assert(!deque_.empty() && "MonotonicQueue::max() called on empty queue");
        return deque_.front().value;
    }

    /**
     * Returns true if the queue is empty (no elements in window)
     */
    bool empty() const { return deque_.empty(); }

    /**
     * Returns the number of elements in the current window
     * Note: This is the logical window size, not the internal candidate count
     */
    size_t size() const {
        return counter_ < window_size_ ? static_cast<size_t>(counter_) : static_cast<size_t>(window_size_);
    }

    /**
     * Returns the number of candidate elements stored internally
     * (May be less than window size due to monotonic property)
     */
    size_t candidate_count() const { return deque_.size(); }

    /**
     * Clears all elements from the queue
     */
    void clear() {
        deque_.clear();
        counter_ = 0;
    }
};
