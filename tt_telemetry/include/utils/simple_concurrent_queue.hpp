#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * utils/simple_concurrent_queue.hpp
 *
 * A simple thread-safe queue wrapper around std::queue with mutex protection.
 * Provides safe concurrent access for multiple producers and single/multiple consumers.
 *
 * Supports both polling (pop) and blocking (pop_wait) patterns:
 * - pop(): Non-blocking, returns immediately with std::nullopt if empty
 * - pop_wait(): Blocks until data is available or timeout/shutdown occurs
 */

#include <queue>
#include <mutex>
#include <optional>
#include <condition_variable>
#include <atomic>
#include <chrono>

template <typename T>
class SimpleConcurrentQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};

public:
    SimpleConcurrentQueue() = default;

    ~SimpleConcurrentQueue() { shutdown(); }

    // Non-copyable, non-movable for simplicity
    SimpleConcurrentQueue(const SimpleConcurrentQueue&) = delete;
    SimpleConcurrentQueue& operator=(const SimpleConcurrentQueue&) = delete;
    SimpleConcurrentQueue(SimpleConcurrentQueue&&) = delete;
    SimpleConcurrentQueue& operator=(SimpleConcurrentQueue&&) = delete;

    /**
     * Add an element to the back of the queue.
     * Thread-safe for multiple producers.
     * Notifies all waiting threads.
     *
     * @param item The item to add to the queue
     */
    void push(const T& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(item);
        }
        cv_.notify_all();
    }

    /**
     * Add an element to the back of the queue (move version).
     * Thread-safe for multiple producers.
     * Notifies all waiting threads.
     *
     * @param item The item to move into the queue
     */
    void push(T&& item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_all();
    }

    /**
     * Remove and return the front element from the queue (non-blocking).
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
     * Remove and return the front element from the queue (blocking, no timeout).
     * Blocks until an item is available or the queue is shut down.
     * Thread-safe for multiple consumers.
     *
     * @return std::optional<T> containing the front element, or std::nullopt if queue is shutdown
     */
    std::optional<T> pop_wait() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has data or is shutdown
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

        // Check if we have data (might have lost race or be shutting down)
        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    /**
     * Remove and return the front element from the queue (blocking with timeout).
     * Blocks until an item is available, timeout expires, or the queue is shut down.
     * Thread-safe for multiple consumers.
     *
     * @param timeout Maximum time to wait for an item
     * @return std::optional<T> containing the front element, or std::nullopt if timeout/shutdown
     */
    template <typename Rep, typename Period>
    std::optional<T> pop_wait(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait until queue has data, timeout, or shutdown
        if (!cv_.wait_for(lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
            return std::nullopt;  // Timeout
        }

        // Check if we have data (might have lost race or be shutting down)
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

    /**
     * Signal shutdown to all waiting threads.
     * All threads blocked in pop_wait() will wake up and return std::nullopt.
     * Thread-safe and idempotent.
     */
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();  // Wake all waiting threads
    }

    /**
     * Check if the queue has been shut down.
     * Thread-safe.
     *
     * @return true if shutdown() has been called, false otherwise
     */
    bool is_shutdown() const { return shutdown_; }
};
