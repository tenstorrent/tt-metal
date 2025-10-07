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

class MockTelemetryProvider {
private:
    static constexpr auto UPDATE_INTERVAL_SECONDS = std::chrono::seconds(5);

    // Telemetry metrics state
    const std::vector<std::string> bool_metric_names_ = {
        "foo/bar/baz1", "foo/bar/baz2", "foo/bar/baz3", "foo/glorp/cpu1", "foo/glorp/cpu2", "foo/glorp/cpu3"};
    std::vector<size_t> bool_metric_ids_;
    std::vector<bool> bool_metric_values_;
    const std::vector<std::string> uint_metric_names_ = {"foo/bar/baz4int", "ints/intermediate/intvalue"};
    const std::vector<MetricUnit> uint_metric_units_ = {MetricUnit::WATTS, MetricUnit::REVOLUTIONS_PER_MINUTE};
    std::vector<size_t> uint_metric_ids_;
    std::vector<uint64_t> uint_metric_values_;

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
            std::cout << "[MockTelemetryProvider] Returned buffer" << std::endl;
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
            std::cout << "[MockTelemetryProvider] Got buffer from pool" << std::endl;
        } else {
            // Pool exhausted, create new buffer
            buffer = new TelemetrySnapshot();
            std::cout << "[MockTelemetryProvider] Allocated new buffer" << std::endl;
        }

        // Return a RAII handle that will automatically return buffer to pool
        return create_new_handoff_buffer(buffer);
    }

    void create_random_updates(std::shared_ptr<TelemetrySnapshot> delta) {
        // Select random bool metrics to update
        std::uniform_int_distribution<> bool_metric_dist(0, bool_metric_names_.size() - 1);
        int num_updates = num_updates_dist_(gen_);
        for (int i = 0; i < num_updates; ++i) {
            size_t idx = bool_metric_dist(gen_);
            size_t id = bool_metric_ids_[idx];
            bool new_value = bool_dist_(gen_) & 1;

            // Only add to updates if state actually changes
            if (bool_metric_values_[idx] != new_value) {
                bool_metric_values_[idx] = new_value;
                delta->bool_metric_ids.push_back(id);
                delta->bool_metric_values.push_back(new_value);
                // Add current timestamp
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                delta->bool_metric_timestamps.push_back(timestamp);
            }
        }

        // Select random uint metrics to update
        std::uniform_int_distribution<> uint_metric_dist(0, uint_metric_names_.size() - 1);
        num_updates = num_updates_dist_(gen_);
        for (int i = 0; i < num_updates; ++i) {
            size_t idx = uint_metric_dist(gen_);
            size_t id = uint_metric_ids_[idx];
            uint64_t new_value = uint_dist_(gen_);

            // Only add to updates if state actually changes
            if (uint_metric_values_[idx] != new_value) {
                uint_metric_values_[idx] = new_value;
                delta->uint_metric_ids.push_back(id);
                delta->uint_metric_values.push_back(new_value);
                // Add current timestamp
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                delta->uint_metric_timestamps.push_back(timestamp);
            }
        }

        std::cout << "[MockTelemetryProvider] Updated: "
                  << (delta->bool_metric_ids.size() + delta->uint_metric_ids.size()) << " values pending" << std::endl;
    }

    void update_telemetry_randomly() {
        // Send initial snapshot to subscriber
        {
            std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
            snapshot->clear();
            for (size_t i = 0; i < bool_metric_names_.size(); i++) {
                size_t id = bool_metric_ids_[i];
                snapshot->bool_metric_ids.push_back(id);
                snapshot->bool_metric_names.push_back(bool_metric_names_[i]);
                snapshot->bool_metric_values.push_back(bool_metric_values_[i]);
                // Add current timestamp for initial snapshot
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                snapshot->bool_metric_timestamps.push_back(timestamp);
            }
            for (size_t i = 0; i < uint_metric_names_.size(); i++) {
                size_t id = uint_metric_ids_[i];
                snapshot->uint_metric_ids.push_back(id);
                snapshot->uint_metric_names.push_back(uint_metric_names_[i]);
                snapshot->uint_metric_units.push_back(static_cast<uint16_t>(uint_metric_units_[i]));
                snapshot->uint_metric_values.push_back(uint_metric_values_[i]);
                // Add current timestamp for initial snapshot
                uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                snapshot->uint_metric_timestamps.push_back(timestamp);
            }

            // Populate unit label maps when names are populated
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
    explicit MockTelemetryProvider(std::initializer_list<std::shared_ptr<TelemetrySubscriber>> subscribers) :
        subscribers_(subscribers), gen_(rd_()), bool_dist_(0, 1), uint_dist_(0, 1000), num_updates_dist_(1, 4) {
        // Init telemetry randomly
        bool_metric_ids_.clear();
        bool_metric_ids_.reserve(bool_metric_names_.size());
        bool_metric_values_.clear();
        bool_metric_values_.reserve(bool_metric_names_.size());
        uint_metric_ids_.clear();
        uint_metric_ids_.reserve(uint_metric_names_.size());
        uint_metric_values_.clear();
        uint_metric_values_.reserve(uint_metric_names_.size());

        size_t id = 1;
        for (size_t i = 0; i < bool_metric_names_.size(); ++i) {
            bool initial_value = bool_dist_(gen_) & 1;
            bool_metric_values_.push_back(initial_value);
            bool_metric_ids_.push_back(id++);
        }
        for (size_t i = 0; i < uint_metric_names_.size(); ++i) {
            uint64_t initial_value = uint_dist_(gen_);
            uint_metric_values_.push_back(initial_value);
            uint_metric_ids_.push_back(id++);
        }
    }

    void run() {
        // Run mock telemetry update thread
        auto t = std::async(std::launch::async, &MockTelemetryProvider::update_telemetry_randomly, this);
        t.wait();
    }
};
