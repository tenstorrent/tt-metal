#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * utils/simple_concurrent_queue.hpp
 *
 * A simple thread-safe queue wrapper around std::queue with mutex protection.
 * Provides safe concurrent access for multiple producers and single/multiple consumers.
 */

#include <queue>
#include <mutex>
#include <optional>

template <typename T>
class SimpleConcurrentQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;

public:
    SimpleConcurrentQueue() = default;
    ~SimpleConcurrentQueue() = default;

    // Non-copyable, non-movable for simplicity
    SimpleConcurrentQueue(const SimpleConcurrentQueue&) = delete;
    SimpleConcurrentQueue& operator=(const SimpleConcurrentQueue&) = delete;
    SimpleConcurrentQueue(SimpleConcurrentQueue&&) = delete;
    SimpleConcurrentQueue& operator=(SimpleConcurrentQueue&&) = delete;

    /**
     * Add an element to the back of the queue.
     * Thread-safe for multiple producers.
     *
     * @param item The item to add to the queue
     */
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
    }

    /**
     * Add an element to the back of the queue (move version).
     * Thread-safe for multiple producers.
     *
     * @param item The item to move into the queue
     */
    void push(T&& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
    }

    /**
     * Remove and return the front element from the queue.
     * Thread-safe for multiple consumers.
     *
     * @return std::optional<T> containing the front element, or std::nullopt if queue is empty
     */
    std::optional<T> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    /**
     * Check if the queue is empty.
     * Thread-safe.
     *
     * @return true if the queue is empty, false otherwise
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * Get the number of elements in the queue.
     * Thread-safe.
     *
     * @return The number of elements in the queue
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /**
     * Process all elements in the queue with a callback function.
     * Thread-safe and atomic - processes all elements that were in the queue
     * at the time of the call.
     *
     * @param callback Function to call for each element: void callback(T&& item)
     */
    template <typename Callback>
    void process_all(Callback callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            callback(std::move(queue_.front()));
            queue_.pop();
        }
    }
};
