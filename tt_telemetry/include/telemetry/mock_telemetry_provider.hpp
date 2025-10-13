#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/mock_telemetry_provider.hpp
 *
 * Generates fake telemetry. Useful for testing.
 *
 * TODO:
 * -----
 * - Mock telemetry should avoid sending to subscribers until they have all finished consuming the
 *   current buffer. It should continue to update the same delta snapshot until it can push it.
 */

#include <atomic>
#include <chrono>
#include <future>
#include <initializer_list>
#include <mutex>
#include <queue>
#include <random>
#include <vector>

#include <telemetry/telemetry_subscriber.hpp>

class MockTelemetryCollector {
private:
    static constexpr auto UPDATE_INTERVAL_SECONDS = std::chrono::seconds(5);

    // Telemetry metrics state - using same structure as TelemetrySnapshot
    std::unordered_map<std::string, bool> bool_metrics_ = {
        {"foo/bar/baz1", false},
        {"foo/bar/baz2", false},
        {"foo/bar/baz3", false},
        {"foo/glorp/cpu1", false},
        {"foo/glorp/cpu2", false},
        {"foo/glorp/cpu3", false}};
    std::unordered_map<std::string, uint64_t> uint_metrics_ = {
        {"foo/bar/baz4int", 0}, {"ints/intermediate/intvalue", 0}};
    const std::unordered_map<std::string, MetricUnit> uint_metric_units_ = {
        {"foo/bar/baz4int", MetricUnit::WATTS}, {"ints/intermediate/intvalue", MetricUnit::REVOLUTIONS_PER_MINUTE}};

    // Snapshot and distribution to consumers
    std::mutex mtx_;
    std::queue<TelemetrySnapshot*> available_buffers_;
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers_;
    std::atomic<bool> stopped_{false};

    // Random number generators for generating random updates
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<> bool_dist_;
    std::uniform_int_distribution<uint64_t> uint_dist_;
    std::uniform_int_distribution<> num_updates_dist_;

    void return_buffer_to_pool(TelemetrySnapshot* buffer) {
        if (stopped_.load()) {
            return;
        }

        // Clear buffer and return to pool
        buffer->clear();
        std::lock_guard<std::mutex> lock(mtx_);
        available_buffers_.push(buffer);
    }

    std::shared_ptr<TelemetrySnapshot> create_new_handoff_buffer(TelemetrySnapshot* buffer) {
        return std::shared_ptr<TelemetrySnapshot>(buffer, [this](TelemetrySnapshot* buffer) {
            // Custom deleter: do not delete, just return to pool. We use shared_ptr for its
            // thread-safe reference counting, allowing a buffer to be passed to multiple
            // consumers.
            std::cout << "[MockTelemetryCollector] Returned buffer" << std::endl;
            return_buffer_to_pool(buffer);
        });
    }

    std::shared_ptr<TelemetrySnapshot> get_writeable_buffer() {
        std::lock_guard<std::mutex> lock(mtx_);

        TelemetrySnapshot* buffer;
        if (!available_buffers_.empty()) {
            // Get a free buffer
            buffer = available_buffers_.front();
            available_buffers_.pop();
            std::cout << "[MockTelemetryCollector] Got buffer from pool" << std::endl;
        } else {
            // Pool exhausted, create new buffer
            buffer = new TelemetrySnapshot();
            std::cout << "[MockTelemetryCollector] Allocated new buffer" << std::endl;
        }

        // Return a RAII handle that will automatically return buffer to pool
        return create_new_handoff_buffer(buffer);
    }

    void create_random_updates(std::shared_ptr<TelemetrySnapshot> delta) {
        // Create a vector of bool metric paths for random selection
        std::vector<std::string> bool_paths;
        for (const auto& [path, value] : bool_metrics_) {
            bool_paths.push_back(path);
        }

        // Select random bool metrics to update
        std::uniform_int_distribution<> bool_metric_dist(0, bool_paths.size() - 1);
        int num_updates = num_updates_dist_(gen_);
        for (int i = 0; i < num_updates; ++i) {
            const std::string& path = bool_paths[bool_metric_dist(gen_)];
            bool new_value = bool_dist_(gen_) & 1;

            // Only add to updates if state actually changes
            if (bool_metrics_[path] != new_value) {
                bool_metrics_[path] = new_value;
                delta->bool_metrics[path] = new_value;
                // Add current timestamp
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                delta->bool_metric_timestamps[path] = timestamp;
            }
        }

        // Create a vector of uint metric paths for random selection
        std::vector<std::string> uint_paths;
        for (const auto& [path, value] : uint_metrics_) {
            uint_paths.push_back(path);
        }

        // Select random uint metrics to update
        std::uniform_int_distribution<> uint_metric_dist(0, uint_paths.size() - 1);
        num_updates = num_updates_dist_(gen_);
        for (int i = 0; i < num_updates; ++i) {
            const std::string& path = uint_paths[uint_metric_dist(gen_)];
            uint64_t new_value = uint_dist_(gen_);

            // Only add to updates if state actually changes
            if (uint_metrics_[path] != new_value) {
                uint_metrics_[path] = new_value;
                delta->uint_metrics[path] = new_value;
                // Add current timestamp
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                delta->uint_metric_timestamps[path] = timestamp;
            }
        }

        std::cout << "[MockTelemetryCollector] Updated: " << (delta->bool_metrics.size() + delta->uint_metrics.size())
                  << " values pending" << std::endl;
    }

    void update_telemetry_randomly() {
        // Send initial snapshot to subscriber
        {
            std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
            snapshot->clear();

            // Copy all bool metrics
            for (const auto& [path, value] : bool_metrics_) {
                snapshot->bool_metrics[path] = value;
                // Add current timestamp for initial snapshot
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                snapshot->bool_metric_timestamps[path] = timestamp;
            }

            // Copy all uint metrics
            for (const auto& [path, value] : uint_metrics_) {
                snapshot->uint_metrics[path] = value;
                snapshot->uint_metric_units[path] = static_cast<uint16_t>(uint_metric_units_.at(path));
                // Add current timestamp for initial snapshot
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                snapshot->uint_metric_timestamps[path] = timestamp;
            }

            // Populate unit label maps for initial snapshot
            snapshot->metric_unit_display_label_by_code = create_metric_unit_display_label_map();
            snapshot->metric_unit_full_label_by_code = create_metric_unit_full_label_map();

            for (auto& subscriber : subscribers_) {
                subscriber->on_telemetry_ready(snapshot);
            }
        }

        // Send periodic updates
        while (!stopped_.load()) {
            // We will now produce a delta snapshot
            std::shared_ptr<TelemetrySnapshot> delta = get_writeable_buffer();
            delta->clear();

            // Fill it with updates
            create_random_updates(delta);

            // Push to subscribers
            for (auto& subscriber : subscribers_) {
                subscriber->on_telemetry_ready(delta);
            }

            std::this_thread::sleep_for(UPDATE_INTERVAL_SECONDS);
        }
    }

public:
    explicit MockTelemetryCollector(const std::vector<std::shared_ptr<TelemetrySubscriber>>& subscribers) :
        subscribers_(subscribers), gen_(rd_()), bool_dist_(0, 1), uint_dist_(0, 1000), num_updates_dist_(1, 4) {
        // Initialize telemetry with random values
        for (auto& [path, value] : bool_metrics_) {
            value = bool_dist_(gen_) & 1;
        }
        for (auto& [path, value] : uint_metrics_) {
            value = uint_dist_(gen_);
        }
    }

    void run() {
        // Run mock telemetry update thread
        auto t = std::async(std::launch::async, &MockTelemetryCollector::update_telemetry_randomly, this);
        t.wait();
    }
};
