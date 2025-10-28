// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The telemetry collector creates a thread that collects and disseminates telemetry. Note that
 * subscribers here are internal components. They maintain their own complete state, therefore
 * the initial snapshot need only be sent once (the subscribers can never "disconnect") and any
 * metrics received from remote instances are propagated via delta updates.
 */

#include <functional>
#include <future>
#include <queue>
#include <unistd.h>

#include <boost/functional/hash.hpp>

#include <tt-logger/tt-logger.hpp>
#include <server/collection_clients.hpp>
#include <utils/simple_concurrent_queue.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>
#include "protobuf/factory_system_descriptor.pb.h"

#include <telemetry/telemetry_collector.hpp>
#include <telemetry/watchdog.hpp>
#include <hal/hal.hpp>
#include <telemetry/ethernet/ethernet_metrics.hpp>
#include <telemetry/arc/arc_metrics.hpp>
#include <topology/topology.hpp>

static constexpr auto MONITOR_INTERVAL_SECONDS = std::chrono::seconds(5);

static std::mutex mtx_;
static std::queue<TelemetrySnapshot*> available_buffers_;

static std::atomic<bool> stopped_{false};

static char hostname_[256];

static std::vector<std::unique_ptr<BoolMetric>> bool_metrics_;
static std::vector<std::unique_ptr<UIntMetric>> uint_metrics_;
static std::vector<std::unique_ptr<DoubleMetric>> double_metrics_;

// Unbounded queue for storing received telemetry snapshots from aggregate endpoints
static SimpleConcurrentQueue<std::pair<std::string, TelemetrySnapshot>> received_snapshots_;

// Callback function for handling received telemetry snapshots from WebSocket clients
static void on_snapshot_received(const std::string& endpoint, const TelemetrySnapshot& snapshot) {
    log_debug(tt::LogAlways, "TelemetryCollector: Received snapshot from endpoint {}", endpoint);

    // Add the received endpoint-snapshot pair to the queue (thread-safe)
    received_snapshots_.push(std::make_pair(endpoint, snapshot));

    log_debug(tt::LogAlways, "TelemetryCollector: Snapshot from {} added to queue", endpoint);
}

static void aggregate_remote_telemetry(TelemetrySnapshot& delta_snapshot) {
    // Process all received snapshots from remote telemetry endpoints
    received_snapshots_.process_all([&delta_snapshot](auto&& endpoint_snapshot_pair) {
        auto [endpoint, snapshot] = std::move(endpoint_snapshot_pair);

        log_debug(tt::LogAlways, "TelemetryCollector: Processing snapshot from endpoint {}", endpoint);

        // Merge the received snapshot into the delta snapshot for this update cycle
        delta_snapshot.merge_from(snapshot);

        log_debug(tt::LogAlways, "TelemetryCollector: Added remote telemetry data from {} to delta", endpoint);
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
    return std::shared_ptr<TelemetrySnapshot>(buffer, [](TelemetrySnapshot* buffer) {
        // Custom deleter: do not delete, just return to pool. We use shared_ptr for its
        // thread-safe reference counting, allowing a buffer to be passed to multiple
        // consumers.
        log_debug(tt::LogAlways, "TelemetryCollector: Returned buffer");
        return_buffer_to_pool(buffer);
    });
}

static std::shared_ptr<TelemetrySnapshot> get_writeable_buffer() {
    std::lock_guard<std::mutex> lock(mtx_);

    TelemetrySnapshot* buffer;
    if (!available_buffers_.empty()) {
        // Get a free buffer
        buffer = available_buffers_.front();
        available_buffers_.pop();
        log_debug(tt::LogAlways, "TelemetryCollector: Got buffer from pool");
    } else {
        // Pool exhausted, create new buffer
        buffer = new TelemetrySnapshot();
        log_debug(tt::LogAlways, "TelemetryCollector: Allocated new buffer");
    }

    // Ensure it is clear
    buffer->clear();

    // Return a RAII handle that will automatically return buffer to pool
    return create_new_handoff_buffer(buffer);
}

static std::string get_cluster_wide_telemetry_path(const Metric& metric) {
    // Cluster-wide path is: hostname + metric path
    std::vector<std::string> path_components{static_cast<const char*>(hostname_)};
    auto local_path = metric.telemetry_path();
    path_components.insert(path_components.end(), local_path.begin(), local_path.end());

    // Join with '/'
    std::string path;
    for (auto it = path_components.begin(); it != path_components.end();) {
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
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_of_update_cycle - start_of_update_cycle).count();
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

    for (auto& subscriber : subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void update_delta_snapshot_with_local_telemetry(std::shared_ptr<TelemetrySnapshot> snapshot) {
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
        snapshot->uint_metric_units[path] = static_cast<uint16_t>(uint_metrics_[i]->units);
        snapshot->uint_metric_timestamps[path] = uint_metrics_[i]->timestamp();
        uint_metrics_[i]->mark_transmitted();
    }

    for (size_t i = 0; i < double_metrics_.size(); i++) {
        if (!double_metrics_[i]->changed_since_transmission()) {
            continue;
        }
        std::string path = get_cluster_wide_telemetry_path(*double_metrics_[i]);
        snapshot->double_metrics[path] = double_metrics_[i]->value();
        snapshot->double_metric_units[path] = static_cast<uint16_t>(double_metrics_[i]->units);
        snapshot->double_metric_timestamps[path] = double_metrics_[i]->timestamp();
        double_metrics_[i]->mark_transmitted();
    }

    // Add unit label maps to the snapshot (if not already present from remote aggregation)
    if (snapshot->metric_unit_display_label_by_code.empty()) {
        snapshot->metric_unit_display_label_by_code = create_metric_unit_display_label_map();
    }
    if (snapshot->metric_unit_full_label_by_code.empty()) {
        snapshot->metric_unit_full_label_by_code = create_metric_unit_full_label_map();
    }
}

static void telemetry_thread(
    bool telemetry_enabled,
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers,
    const std::vector<std::string>& aggregate_endpoints,
    const tt::llrt::RunTimeOptions& rtoptions,
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd,
    int watchdog_timeout_seconds) {
    try {
        Watchdog watchdog(watchdog_timeout_seconds);
        TT_FATAL(
            watchdog_timeout_seconds <= 0 || std::chrono::seconds(watchdog_timeout_seconds) > MONITOR_INTERVAL_SECONDS,
            "Watchdog timeout ({} seconds) cannot be shorter than the telemetry monitoring interval ({} seconds)",
            watchdog_timeout_seconds,
            MONITOR_INTERVAL_SECONDS.count());

        std::unique_ptr<tt::umd::Cluster> cluster;
        std::unique_ptr<tt::tt_metal::Hal> hal;
        std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> psd;
        std::unique_ptr<TopologyHelper> topology_translation;

        if (telemetry_enabled) {
            cluster = std::make_unique<tt::umd::Cluster>();
            auto distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
            hal = create_hal(cluster);
            psd = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
                cluster, distributed_context, hal.get(), rtoptions);
            topology_translation = std::make_unique<TopologyHelper>(cluster, psd);
            log_info(tt::LogAlways, "Created cluster, physical system descriptor, and HAL");
            log_info(tt::LogAlways, "Our hostname is: {}", topology_translation->my_host_name);

            create_ethernet_metrics(
                bool_metrics_, uint_metrics_, double_metrics_, cluster, fsd, topology_translation, hal);
            create_arc_metrics(bool_metrics_, uint_metrics_, double_metrics_, cluster, topology_translation, hal);
            log_info(tt::LogAlways, "Initialized metrics");
        }

        // Create collection clients to connect to aggregate endpoints (if any specified)
        CollectionClients collection_clients(aggregate_endpoints, on_snapshot_received);
        log_info(tt::LogAlways, "Collection clients created successfully");

        // Get initial telemetry reading
        if (telemetry_enabled) {
            update(cluster);
            send_initial_snapshot(subscribers);
            log_info(tt::LogAlways, "Obtained initial readout and sent snapshot");
        }

        // Increment heartbeat after initialization
        watchdog.heartbeat();

        // Main telemetry monitoring loop
        while (!stopped_.load()) {
            try {
                std::this_thread::sleep_for(MONITOR_INTERVAL_SECONDS);

                if (telemetry_enabled) {
                    // Collect local telemetry
                    update(cluster);
                }

                // Aggregate from remotes and produce delta snapshot
                std::shared_ptr<TelemetrySnapshot> delta_snapshot = get_writeable_buffer();
                aggregate_remote_telemetry(*delta_snapshot);
                update_delta_snapshot_with_local_telemetry(delta_snapshot);

                // Send to subscribers
                for (auto& subscriber : subscribers) {
                    subscriber->on_telemetry_ready(delta_snapshot);
                }

                // Increment heartbeat to signal watchdog that we've completed this loop iteration
                watchdog.heartbeat();

            } catch (const std::exception& e) {
                log_fatal(tt::LogAlways, "Exception in telemetry monitoring loop: {}", e.what());
            } catch (...) {
                log_fatal(tt::LogAlways, "Unknown exception in telemetry monitoring loop");
            }
        }
    } catch (const std::exception& e) {
        log_fatal(tt::LogAlways, "Fatal exception during telemetry thread initialization: {}", e.what());
    } catch (...) {
        log_fatal(tt::LogAlways, "Unknown fatal exception during telemetry thread initialization");
    }

    // Telemetry thread should currently never stop. If it happens, it's an error and should terminate the app.
    log_fatal(tt::LogAlways, "Telemetry thread stopped");
    exit(1);  // kill process so that job scheduler notices and restarts app
}

void run_telemetry_collector(
    bool telemetry_enabled,
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers,
    const std::vector<std::string>& aggregate_endpoints,
    const tt::llrt::RunTimeOptions& rtoptions,
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd,
    int watchdog_timeout_seconds) {
    // Prefill hostname
    gethostname(hostname_, sizeof(hostname_));

    // Run telemetry thread
    auto t = std::async(
        std::launch::async,
        telemetry_thread,
        telemetry_enabled,
        subscribers,
        aggregate_endpoints,
        std::cref(rtoptions),
        fsd,
        watchdog_timeout_seconds);
    t.wait();
}
