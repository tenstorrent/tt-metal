#pragma once

/*
 * server/mock_telemetry_provider.hpp
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
 #include <initializer_list>
 #include <mutex>
 #include <queue>
 #include <random>
 #include <thread>
 #include <vector>

 #include <server/telemetry_subscriber.hpp>

 class MockTelemetryProvider {
    private:
        static constexpr auto UPDATE_INTERVAL_SECONDS = std::chrono::seconds(5);
    
        // Telemetry metrics state
        const std::vector<std::string> bool_metric_names_ = {
            "foo_bar_baz1",
            "foo_bar_baz2",
            "foo_bar_baz3",
            "foo_glorp_cpu1",
            "foo_glorp_cpu2",
            "foo_glorp_cpu3"
        };
        std::vector<bool> bool_metric_values_;
        const std::vector<std::string> int_metric_names_ = {
            "foo_bar_baz4int",
            "ints_intermediate_intvalue"
        };
        std::vector<int> int_metric_values_;
    
        // Snapshot and distribution to consumers
        std::mutex mtx_;
        std::queue<TelemetrySnapshot *> available_buffers_;
        std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers_;
        std::atomic<bool> stopped_{false};
    
        // Random number generators for generating random updates
        std::random_device rd_;
        std::mt19937 gen_;
        std::uniform_int_distribution<> bool_dist_;
        std::uniform_int_distribution<> int_dist_;
        std::uniform_int_distribution<> metric_dist_;
        std::uniform_int_distribution<> num_updates_dist_;
    
        std::thread thread_;
    
        void return_buffer_to_pool(TelemetrySnapshot *buffer) {
            if (stopped_.load()) {
                return;
            }
    
            // Clear buffer and return to pool
            buffer->clear();
            std::lock_guard<std::mutex> lock(mtx_);
            available_buffers_.push(buffer);
        }
    
        std::shared_ptr<TelemetrySnapshot> create_new_handoff_buffer(TelemetrySnapshot *buffer) {
            return std::shared_ptr<TelemetrySnapshot>(
                buffer,
                [this](TelemetrySnapshot *buffer) {
                    // Custom deleter: do not delete, just return to pool. We use shared_ptr for its
                    // thread-safe reference counting, allowing a buffer to be passed to multiple
                    // consumers.
                    std::cout << "[MockTelemetryProvider] Released buffer" << std::endl;
                    return_buffer_to_pool(buffer);
                }
            );
        }
    
        std::shared_ptr<TelemetrySnapshot> get_writeable_buffer() {
            std::lock_guard<std::mutex> lock(mtx_);
    
            TelemetrySnapshot *buffer;
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
            int num_updates = num_updates_dist_(gen_);
            for (int i = 0; i < num_updates; ++i) {
                size_t idx = metric_dist_(gen_);
                bool new_value = bool_dist_(gen_) & 1;
                
                // Only add to updates if state actually changes
                if (bool_metric_values_[idx] != new_value) {
                    bool_metric_values_[idx] = new_value;
                    delta->bool_metric_indices.push_back(idx);
                    delta->bool_metric_values.push_back(new_value);
                }
            }

            // Select random bool metrics to update
            num_updates = num_updates_dist_(gen_);
            for (int i = 0; i < num_updates; ++i) {
                size_t idx = metric_dist_(gen_);
                bool new_value = int_dist_(gen_);
                
                // Only add to updates if state actually changes
                if (int_metric_values_[idx] != new_value) {
                    int_metric_values_[idx] = new_value;
                    delta->int_metric_indices.push_back(idx);
                    delta->int_metric_values.push_back(new_value);
                }
            }
    
            std::cout << "[MockTelemetryProvider] Updated: " << (delta->bool_metric_indices.size() + delta->int_metric_indices.size()) << " values pending" << std::endl;
        }
    
        void update_telemetry_randomly() {
            // Send initial snapshot to subscriber
            {
                std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
                snapshot->clear();
                snapshot->is_absolute = true;
                for (size_t i = 0; i < bool_metric_names_.size(); i++) {
                    snapshot->bool_metric_indices.push_back(i);
                    snapshot->bool_metric_names.push_back(bool_metric_names_[i]);
                    snapshot->bool_metric_values.push_back(bool_metric_values_[i]);
                }
                for (size_t i = 0; i < int_metric_names_.size(); i++) {
                    snapshot->int_metric_indices.push_back(i);
                    snapshot->int_metric_names.push_back(int_metric_names_[i]);
                    snapshot->int_metric_values.push_back(int_metric_values_[i]);
                }
    
                for (auto &subscriber: subscribers_) {
                    subscriber->on_telemetry_ready(snapshot);
                }
            }
    
            // Send periodic updates
            while (!stopped_.load()) {
                // We will now produce a delta snapshot
                std::shared_ptr<TelemetrySnapshot> delta = get_writeable_buffer();
                delta->clear();
                delta->is_absolute = false;
    
                // Fill it with updates
                create_random_updates(delta);
    
                // Push to subscribers
                for (auto &subscriber: subscribers_) {
                    subscriber->on_telemetry_ready(delta);
                }
    
                std::this_thread::sleep_for(UPDATE_INTERVAL_SECONDS);
            }
        }
    
    public:
        explicit MockTelemetryProvider(std::initializer_list<std::shared_ptr<TelemetrySubscriber>> subscribers)
            : subscribers_(subscribers)
            , gen_(rd_())
            , bool_dist_(0, 1)
            , int_dist_(-100, 100)
            , metric_dist_(0, bool_metric_names_.size() - 1)
            , num_updates_dist_(1, 4) {
            // Init telemetry randomly
            bool_metric_values_.clear();
            bool_metric_values_.reserve(bool_metric_names_.size());
            int_metric_values_.clear();
            int_metric_values_.reserve(int_metric_names_.size());
            
            for (size_t i = 0; i < bool_metric_names_.size(); ++i) {
                bool initial_value = bool_dist_(gen_) & 1;
                bool_metric_values_.push_back(initial_value);
            }
            for (size_t i = 0; i < int_metric_names_.size(); ++i) {
                int initial_value = int_dist_(gen_);
                int_metric_values_.push_back(initial_value);
            }
    
            // Start update thread
            thread_ = std::thread(&MockTelemetryProvider::update_telemetry_randomly, this);
        }
    };