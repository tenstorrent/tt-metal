// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <queue>
#include <unistd.h>

#include <boost/functional/hash.hpp>

#include <tt-logger/tt-logger.hpp>
#include <server/websocket_clients.hpp>
#include <utils/simple_concurrent_queue.hpp>

#include <telemetry/telemetry_provider.hpp>
#include <hal/hal.hpp>
#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/arc/arc_metrics.hpp>
#include <telemetry/arc/arc_telemetry_reader.hpp>

static constexpr auto MONITOR_INTERVAL_SECONDS = std::chrono::seconds(5);

static std::mutex mtx_;
static std::queue<TelemetrySnapshot *> available_buffers_;

static std::atomic<bool> stopped_{false};

static char hostname_[256];

static std::vector<std::unique_ptr<BoolMetric>> bool_metrics_;
static std::vector<std::unique_ptr<UIntMetric>> uint_metrics_;
static std::vector<std::unique_ptr<DoubleMetric>> double_metrics_;

// Only when aggregating: maps metric IDs of a given host to the global (cluster-wide) space
// of this tt_telemetry instance.
struct StringAndSizeTPairHash {
    size_t operator()(const std::pair<std::string, size_t>& p) const {
        size_t seed = 0;
        boost::hash_combine(seed, p.first);
        boost::hash_combine(seed, p.second);
        return seed;
    }
};
static std::unordered_map<std::pair<std::string, size_t>, size_t, StringAndSizeTPairHash>
    endpoint_and_local_id_to_global_id_;

// Only when aggregating: accumulated state of all remote tt_telemetry instances
static TelemetrySnapshot remote_telemetry_state;

// Unbounded queue for storing received telemetry snapshots from aggregate endpoints
static SimpleConcurrentQueue<std::pair<std::string, TelemetrySnapshot>> received_snapshots_;

// Callback function for handling received telemetry snapshots from WebSocket clients
static void on_snapshot_received(const std::string& endpoint, const TelemetrySnapshot& snapshot) {
    log_debug(tt::LogAlways, "TelemetryProvider: Received snapshot from endpoint {}", endpoint);

    // Add the received endpoint-snapshot pair to the queue (thread-safe)
    received_snapshots_.push(std::make_pair(endpoint, snapshot));

    log_debug(tt::LogAlways, "TelemetryProvider: Snapshot from {} added to queue", endpoint);
}

static void aggregate_remote_telemetry() {
    // Process all received snapshots from remote telemetry endpoints
    received_snapshots_.process_all([](auto&& endpoint_snapshot_pair) {
        auto [endpoint, snapshot] = std::move(endpoint_snapshot_pair);

        log_debug(tt::LogAlways, "TelemetryProvider: Processing snapshot from endpoint {}", endpoint);

        // Merge the received snapshot into remote telemetry state
        remote_telemetry_state.merge_from(snapshot);

        log_debug(tt::LogAlways, "TelemetryProvider: Updated remote telemetry store with data from {}", endpoint);
    });
}

static void return_buffer_to_pool(TelemetrySnapshot* buffer) {
    if (stopped_.load()) {
        return;
    }

    // Clear buffer and return to pool
    buffer->clear();
    std::lock_guard<std::mutex> lock(mtx_);
    available_buffers_.push(buffer);
}

static std::shared_ptr<TelemetrySnapshot> create_new_handoff_buffer(TelemetrySnapshot* buffer) {
    return std::shared_ptr<TelemetrySnapshot>(
        buffer,
        [](TelemetrySnapshot *buffer) {
            // Custom deleter: do not delete, just return to pool. We use shared_ptr for its
            // thread-safe reference counting, allowing a buffer to be passed to multiple
            // consumers.
            log_debug(tt::LogAlways, "TelemetryProvider: Returned buffer");
            return_buffer_to_pool(buffer);
        }
    );
}

static std::shared_ptr<TelemetrySnapshot> get_writeable_buffer() {
    std::lock_guard<std::mutex> lock(mtx_);

    TelemetrySnapshot *buffer;
    if (!available_buffers_.empty()) {
        // Get a free buffer
        buffer = available_buffers_.front();
        available_buffers_.pop();
        log_debug(tt::LogAlways, "TelemetryProvider: Got buffer from pool");
    } else {
        // Pool exhausted, create new buffer
        buffer = new TelemetrySnapshot();
        log_debug(tt::LogAlways, "TelemetryProvider: Allocated new buffer");
    }

    // Ensure it is clear
    buffer->clear();

    // Return a RAII handle that will automatically return buffer to pool
    return create_new_handoff_buffer(buffer);
}

static std::string get_cluster_wide_telemetry_path(const Metric &metric) {
    // Cluster-wide path is: hostname + metric path
    std::vector<std::string> path_components{static_cast<const char *>(hostname_)};
    auto local_path = metric.telemetry_path();
    path_components.insert(path_components.end(), local_path.begin(), local_path.end());

    // Join with '/'
    std::string path;
    for (auto it = path_components.begin(); it != path_components.end(); ) {
        path += *it;
        ++it;
        if (it != path_components.end()) {
            path += '/';
        }
    }
    return path;
}

static void update(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    log_info(tt::LogAlways, "Starting telemetry readout...");
    std::chrono::steady_clock::time_point start_of_update_cycle = std::chrono::steady_clock::now();

    for (auto& metric : bool_metrics_) {
        metric->update(cluster, start_of_update_cycle);
    }

    for (auto& metric : uint_metrics_) {
        metric->update(cluster, start_of_update_cycle);
    }

    for (auto& metric : double_metrics_) {
        metric->update(cluster, start_of_update_cycle);
    }

    std::chrono::steady_clock::time_point end_of_update_cycle = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_of_update_cycle - start_of_update_cycle).count();
    log_info(tt::LogAlways, "Telemetry readout took {} ms", duration_ms);
}

static void send_initial_snapshot(const std::vector<std::shared_ptr<TelemetrySubscriber>>& subscribers) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();

    for (size_t i = 0; i < bool_metrics_.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*bool_metrics_[i]);
        snapshot->bool_metrics[path] = bool_metrics_[i]->value();
        snapshot->bool_metric_timestamps[path] = bool_metrics_[i]->timestamp();
    }

    for (size_t i = 0; i < uint_metrics_.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*uint_metrics_[i]);
        snapshot->uint_metrics[path] = uint_metrics_[i]->value();
        snapshot->uint_metric_units[path] = static_cast<uint16_t>(uint_metrics_[i]->units);
        snapshot->uint_metric_timestamps[path] = uint_metrics_[i]->timestamp();
    }

    for (size_t i = 0; i < double_metrics_.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*double_metrics_[i]);
        snapshot->double_metrics[path] = double_metrics_[i]->value();
        snapshot->double_metric_units[path] = static_cast<uint16_t>(double_metrics_[i]->units);
        snapshot->double_metric_timestamps[path] = double_metrics_[i]->timestamp();
    }

    // Populate unit label maps for initial snapshot
    snapshot->metric_unit_display_label_by_code = create_metric_unit_display_label_map();
    snapshot->metric_unit_full_label_by_code = create_metric_unit_full_label_map();

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void send_delta(const std::vector<std::shared_ptr<TelemetrySubscriber>>& subscribers) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();

    for (size_t i = 0; i < bool_metrics_.size(); i++) {
        if (!bool_metrics_[i]->changed_since_transmission()) {
            continue;
        }
        std::string path = get_cluster_wide_telemetry_path(*bool_metrics_[i]);
        snapshot->bool_metrics[path] = bool_metrics_[i]->value();
        snapshot->bool_metric_timestamps[path] = bool_metrics_[i]->timestamp();
        bool_metrics_[i]->mark_transmitted();
    }

    for (size_t i = 0; i < uint_metrics_.size(); i++) {
        if (!uint_metrics_[i]->changed_since_transmission()) {
            continue;
        }
        std::string path = get_cluster_wide_telemetry_path(*uint_metrics_[i]);
        snapshot->uint_metrics[path] = uint_metrics_[i]->value();
        snapshot->uint_metric_timestamps[path] = uint_metrics_[i]->timestamp();
        uint_metrics_[i]->mark_transmitted();
    }

    for (size_t i = 0; i < double_metrics_.size(); i++) {
        if (!double_metrics_[i]->changed_since_transmission()) {
            continue;
        }
        std::string path = get_cluster_wide_telemetry_path(*double_metrics_[i]);
        snapshot->double_metrics[path] = double_metrics_[i]->value();
        snapshot->double_metric_timestamps[path] = double_metrics_[i]->timestamp();
        double_metrics_[i]->mark_transmitted();
    }

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void telemetry_thread(
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers,
    const std::vector<std::string>& aggregate_endpoints) {
    std::unique_ptr<tt::umd::Cluster> cluster = std::make_unique<tt::umd::Cluster>();
    std::unique_ptr<tt::tt_metal::Hal> hal = create_hal(cluster);
    log_info(tt::LogAlways, "Created cluster and HAL");

    // Create vectors of all metrics we will monitor by value type
    create_ethernet_metrics(bool_metrics_, uint_metrics_, double_metrics_, cluster, hal);
    create_arc_metrics(bool_metrics_, uint_metrics_, double_metrics_, cluster, hal);
    log_info(tt::LogAlways, "Initialized telemetry thread");

    // Create WebSocket clients to connect to aggregate endpoints
    // Always create WebSocketClients - it will handle empty endpoints internally
    WebSocketClients websocket_clients(aggregate_endpoints, on_snapshot_received);
    log_info(tt::LogAlways, "WebSocket clients created successfully");

    // Continuously monitor on a loop
    update(cluster);
    send_initial_snapshot(subscribers);
    log_info(tt::LogAlways, "Obtained initial readout and sent snapshot");
    while (!stopped_.load()) {
        std::this_thread::sleep_for(MONITOR_INTERVAL_SECONDS);
        update(cluster);
        aggregate_remote_telemetry();
        send_delta(subscribers);
    }

    log_info(tt::LogAlways, "Telemetry thread stopped");
}

void run_telemetry_provider(
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers,
    const std::vector<std::string>& aggregate_endpoints) {
    // Prefill hostname
    gethostname(hostname_, sizeof(hostname_));

    // Run telemetry thread
    auto t = std::async(std::launch::async, telemetry_thread, subscribers, aggregate_endpoints);
    t.wait();
}
