// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The telemetry collector creates a thread that collects and disseminates telemetry. Note that
 * subscribers here are internal components. They maintain their own complete state, therefore
 * the initial snapshot need only be sent once (the subscribers can never "disconnect") and any
 * metrics received from remote instances are propagated via delta updates.
 */

#include <algorithm>
#include <array>
#include <functional>
#include <future>
#include <optional>
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
#include <telemetry/system/system_metrics.hpp>
#include <topology/topology.hpp>

static constexpr auto MONITOR_INTERVAL_SECONDS = std::chrono::seconds(5);

// Helper function to format a future time point as a readable string
static std::string format_end_time(std::chrono::seconds duration_from_now) {
    auto end_time_system = std::chrono::system_clock::now() + duration_from_now;
    auto end_time_t = std::chrono::system_clock::to_time_t(end_time_system);
    std::tm tm_buf{};
    if (!localtime_r(&end_time_t, &tm_buf)) {
        // Fallback to UTC if local time conversion fails
        if (!gmtime_r(&end_time_t, &tm_buf)) {
            return "unknown time";
        }
    }
    std::array<char, 32> time_str{};
    std::strftime(time_str.data(), time_str.size(), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(time_str.data());
}

static std::mutex mtx_;
static std::queue<TelemetrySnapshot*> available_buffers_;

static std::atomic<bool> stopped_{false};

static char hostname_[256];

static std::vector<std::unique_ptr<BoolMetric>> bool_metrics_;
static std::vector<std::unique_ptr<UIntMetric>> uint_metrics_;
static std::vector<std::unique_ptr<DoubleMetric>> double_metrics_;
static std::vector<std::unique_ptr<StringMetric>> string_metrics_;

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
    const auto& local_path = metric.telemetry_path();
    path_components.insert(path_components.end(), local_path.begin(), local_path.end());

    // Join with '/' using more efficient approach
    // Pre-calculate total size to avoid reallocations
    size_t total_size = path_components.size() - 1;  // separators
    for (const auto& component : path_components) {
        total_size += component.length();
    }

    std::string path;
    path.reserve(total_size);

    bool first = true;
    for (const auto& component : path_components) {
        if (!first) {
            path += '/';
        }
        path += component;
        first = false;
    }
    return path;
}

// Structure to track metric timing
struct MetricTiming {
    std::string metric_type;
    std::string metric_path;
    int64_t duration_ms;
};

// Global vector to track all metric timings for reporting
static std::vector<MetricTiming> metric_timings_;
static std::mutex metric_timings_mutex_;

// Helper function to update a collection of metrics with exception handling
template <typename MetricType>
static size_t update_metrics_with_exception_handling(
    std::vector<std::unique_ptr<MetricType>>& metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle,
    std::string_view metric_type_name) {
    size_t failed_count = 0;
    constexpr int64_t SLOW_METRIC_THRESHOLD_MS = 100;  // Lowered from 500ms to 100ms

    for (auto& metric : metrics) {
        try {
            auto metric_start = std::chrono::steady_clock::now();
            metric->update(cluster, start_of_update_cycle);
            auto metric_end = std::chrono::steady_clock::now();

            auto metric_duration_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(metric_end - metric_start).count();

            // Track all metric timings for summary reporting
            {
                std::lock_guard<std::mutex> lock(metric_timings_mutex_);
                metric_timings_.push_back(
                    {std::string(metric_type_name), metric->telemetry_path_string(), metric_duration_ms});
            }

            if (metric_duration_ms >= SLOW_METRIC_THRESHOLD_MS) {
                log_warning(
                    tt::LogAlways,
                    "SLOW METRIC UPDATE: {} metric {} took {} ms",
                    metric_type_name,
                    metric->telemetry_path_string(),
                    metric_duration_ms);
            }
        } catch (const std::exception& e) {
            failed_count++;
            log_debug(
                tt::LogAlways,
                "Failed to update {} metric {} (will skip): {}",
                metric_type_name,
                metric->telemetry_path_string(),
                e.what());
        }
    }

    return failed_count;
}

static void update(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    log_info(tt::LogAlways, "Starting telemetry readout...");
    std::chrono::steady_clock::time_point start_of_update_cycle = std::chrono::steady_clock::now();

    // Clear previous timing data
    {
        std::lock_guard<std::mutex> lock(metric_timings_mutex_);
        metric_timings_.clear();
    }

    // Track failed metrics to report summary at end of cycle
    size_t failed_metrics = 0;

    auto bool_start = std::chrono::steady_clock::now();
    failed_metrics += update_metrics_with_exception_handling(bool_metrics_, cluster, start_of_update_cycle, "bool");
    auto bool_end = std::chrono::steady_clock::now();
    auto bool_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(bool_end - bool_start).count();

    auto uint_start = std::chrono::steady_clock::now();
    failed_metrics += update_metrics_with_exception_handling(uint_metrics_, cluster, start_of_update_cycle, "uint");
    auto uint_end = std::chrono::steady_clock::now();
    auto uint_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(uint_end - uint_start).count();

    auto double_start = std::chrono::steady_clock::now();
    failed_metrics += update_metrics_with_exception_handling(double_metrics_, cluster, start_of_update_cycle, "double");
    auto double_end = std::chrono::steady_clock::now();
    auto double_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(double_end - double_start).count();

    auto string_start = std::chrono::steady_clock::now();
    failed_metrics += update_metrics_with_exception_handling(string_metrics_, cluster, start_of_update_cycle, "string");
    auto string_end = std::chrono::steady_clock::now();
    auto string_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(string_end - string_start).count();

    if (failed_metrics > 0) {
        log_warning(
            tt::LogAlways,
            "Skipped {} metrics due to read failures (likely device busy or inaccessible)",
            failed_metrics);
    }

    std::chrono::steady_clock::time_point end_of_update_cycle = std::chrono::steady_clock::now();
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_of_update_cycle - start_of_update_cycle).count();

    // Log metric counts
    log_info(
        tt::LogAlways,
        "Telemetry readout took {} ms (bool: {} ms [{} metrics], uint: {} ms [{} metrics], double: {} ms [{} metrics], "
        "string: {} ms [{} metrics])",
        duration_ms,
        bool_duration_ms,
        bool_metrics_.size(),
        uint_duration_ms,
        uint_metrics_.size(),
        double_duration_ms,
        double_metrics_.size(),
        string_duration_ms,
        string_metrics_.size());

    // Sort metric timings by duration (descending) and report top 10 slowest
    {
        std::lock_guard<std::mutex> lock(metric_timings_mutex_);
        std::sort(metric_timings_.begin(), metric_timings_.end(), [](const MetricTiming& a, const MetricTiming& b) {
            return a.duration_ms > b.duration_ms;
        });

        size_t top_n = std::min(static_cast<size_t>(10), metric_timings_.size());
        if (top_n > 0) {
            log_info(tt::LogAlways, "Top {} slowest metrics:", top_n);
            for (size_t i = 0; i < top_n; ++i) {
                const auto& timing = metric_timings_[i];
                log_info(
                    tt::LogAlways,
                    "  #{}: {} ms - {} metric: {}",
                    i + 1,
                    timing.duration_ms,
                    timing.metric_type,
                    timing.metric_path);
            }
        }
    }
}


// Constants for Ethernet metric path validation
static constexpr size_t TRAY_PREFIX_LEN = 4;     // "tray".length()
static constexpr size_t CHIP_PREFIX_LEN = 4;     // "chip".length()
static constexpr size_t CHANNEL_PREFIX_LEN = 7;  // "channel".length()

// Helper: Check if a telemetry path represents an Ethernet metric
// Ethernet metrics have paths like: tray{n}/chip{m}/channel{l}/metricName
static bool is_ethernet_metric_path(const std::vector<std::string>& path) {
    if (path.size() < 4) {
        return false;
    }

    // Helper to validate prefix and ensure entire suffix is numeric
    auto is_valid_component = [](std::string_view component, std::string_view prefix) {
        if (component.length() <= prefix.length() || component.rfind(prefix, 0) != 0) {
            return false;
        }
        // Validate all characters after prefix are digits
        return std::all_of(
            component.begin() + prefix.length(), component.end(), [](unsigned char c) { return std::isdigit(c); });
    };

    return is_valid_component(path[0], "tray") && is_valid_component(path[1], "chip") &&
           is_valid_component(path[2], "channel");
}

// Helper: Parse EthernetEndpoint from telemetry path components
// Returns nullopt if parsing fails
static std::optional<EthernetEndpoint> parse_ethernet_endpoint(const std::vector<std::string>& path) {
    // Validate path has enough components and they're long enough
    // Must match is_ethernet_metric_path requirement of >= 4 components
    if (path.size() < 4 || path[0].length() <= TRAY_PREFIX_LEN || path[1].length() <= CHIP_PREFIX_LEN ||
        path[2].length() <= CHANNEL_PREFIX_LEN) {
        return std::nullopt;
    }

    try {
        uint32_t tray_id = std::stoul(path[0].substr(TRAY_PREFIX_LEN));
        uint32_t asic_location = std::stoul(path[1].substr(CHIP_PREFIX_LEN));
        uint32_t channel = std::stoul(path[2].substr(CHANNEL_PREFIX_LEN));

        return EthernetEndpoint{tt::tt_metal::TrayID(tray_id), tt::tt_metal::ASICLocation(asic_location), channel};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// Helper: Convert PhysicalLinkInfo to JSON
static nlohmann::json physical_link_info_to_json(const PhysicalLinkInfo& link_info) {
    nlohmann::json link_json;
    link_json["port_type"] = static_cast<int>(link_info.port_type);
    link_json["port_id"] = *link_info.port_id;
    link_json["is_external"] = link_info.is_external();  // Method call, not member access

    if (link_info.remote_endpoint.has_value()) {
        const auto& remote = link_info.remote_endpoint.value();
        link_json["remote_hostname"] = remote.hostname;
        link_json["remote_tray"] = *remote.tray;
        link_json["remote_asic"] = *remote.asic;
        link_json["remote_channel"] = remote.channel;
        link_json["remote_aisle"] = remote.aisle;
        link_json["remote_rack"] = remote.rack;
    }

    return link_json;
}

// Helper: Populate physical link info for a single metric
static void populate_physical_link_info_for_metric(
    std::string_view metric_path,
    const std::vector<std::string>& telemetry_path,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    std::unordered_map<std::string, nlohmann::json>& physical_link_info_map) {
    if (!topology_translation) {
        return;
    }

    if (!is_ethernet_metric_path(telemetry_path)) {
        return;
    }

    auto endpoint_opt = parse_ethernet_endpoint(telemetry_path);
    if (!endpoint_opt) {
        log_warning(
            tt::LogAlways,
            "Failed to parse Ethernet endpoint from metric path '{}' - numeric conversion error (e.g., overflow)",
            metric_path);
        return;
    }

    auto link_info_opt = topology_translation->get_physical_link_info(endpoint_opt.value());
    if (link_info_opt) {
        physical_link_info_map[std::string(metric_path)] = physical_link_info_to_json(link_info_opt.value());
    }
}

// Template helper: Populate physical link info for all metrics of a given type
template <typename MetricType>
static void populate_physical_link_info_for_metrics(
    const std::vector<std::unique_ptr<MetricType>>& metrics,
    const std::unique_ptr<TopologyHelper>& topology_translation,
    std::unordered_map<std::string, nlohmann::json>& physical_link_info_map) {
    for (const auto& metric : metrics) {
        std::string path = get_cluster_wide_telemetry_path(*metric);
        populate_physical_link_info_for_metric(
            path, metric->telemetry_path(), topology_translation, physical_link_info_map);
    }
}

// Template helper: Process metrics and add them to snapshot
// Uses if constexpr to eliminate branching for the check_changed parameter
// Labels are only sent in initial snapshot (CheckChanged=false) since they are immutable
template <typename MetricType, typename ValueType, bool CheckChanged = false>
static void process_metrics_to_snapshot(
    const std::vector<std::unique_ptr<MetricType>>& metrics,
    std::unordered_map<std::string, ValueType>& value_map,
    std::unordered_map<std::string, uint64_t>& timestamp_map,
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& label_map,
    std::unordered_map<std::string, uint16_t>* unit_map = nullptr) {
    for (const auto& metric : metrics) {
        if constexpr (CheckChanged) {
            // Delta snapshot - only send changed values, no labels (immutable)
            if (!metric->changed_since_transmission()) {
                continue;
            }

            std::string path = get_cluster_wide_telemetry_path(*metric);
            value_map[path] = metric->value();
            timestamp_map[path] = metric->timestamp();
            if (unit_map) {
                (*unit_map)[path] = static_cast<uint16_t>(metric->units);
            }

            metric->mark_transmitted();
        } else {
            // Initial snapshot - send everything including labels
            std::string path = get_cluster_wide_telemetry_path(*metric);
            value_map[path] = metric->value();
            timestamp_map[path] = metric->timestamp();
            if (unit_map) {
                (*unit_map)[path] = static_cast<uint16_t>(metric->units);
            }
            // Get labels from virtual method (may construct on-demand)
            auto labels = metric->labels();
            if (!labels.empty()) {
                label_map[path] = std::move(labels);
            }
        }
    }
}

static void send_initial_snapshot(
    const std::vector<std::shared_ptr<TelemetrySubscriber>>& subscribers,
    const std::unique_ptr<TopologyHelper>& topology_translation) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();

    // Use template helper to process metrics
    process_metrics_to_snapshot<BoolMetric, bool>(
        bool_metrics_, snapshot->bool_metrics, snapshot->bool_metric_timestamps, snapshot->metric_labels);
    process_metrics_to_snapshot<UIntMetric, uint64_t>(
        uint_metrics_,
        snapshot->uint_metrics,
        snapshot->uint_metric_timestamps,
        snapshot->metric_labels,
        &snapshot->uint_metric_units);
    process_metrics_to_snapshot<DoubleMetric, double>(
        double_metrics_,
        snapshot->double_metrics,
        snapshot->double_metric_timestamps,
        snapshot->metric_labels,
        &snapshot->double_metric_units);
    process_metrics_to_snapshot<StringMetric, std::string>(
        string_metrics_,
        snapshot->string_metrics,
        snapshot->string_metric_timestamps,
        snapshot->metric_labels,
        &snapshot->string_metric_units);

    // Populate unit label maps for initial snapshot
    snapshot->metric_unit_display_label_by_code = create_metric_unit_display_label_map();
    snapshot->metric_unit_full_label_by_code = create_metric_unit_full_label_map();

    // Populate physical link info for Ethernet metrics (if topology_translation is available)
    // topology_translation is null when telemetry is disabled (telemetry_enabled=false)
    if (topology_translation) {
        populate_physical_link_info_for_metrics(bool_metrics_, topology_translation, snapshot->physical_link_info);
        populate_physical_link_info_for_metrics(uint_metrics_, topology_translation, snapshot->physical_link_info);
        populate_physical_link_info_for_metrics(double_metrics_, topology_translation, snapshot->physical_link_info);
    }

    for (auto& subscriber : subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void update_delta_snapshot_with_local_telemetry(std::shared_ptr<TelemetrySnapshot> snapshot) {
    // Use template helper to process only changed metrics
    process_metrics_to_snapshot<BoolMetric, bool, true>(
        bool_metrics_, snapshot->bool_metrics, snapshot->bool_metric_timestamps, snapshot->metric_labels);
    process_metrics_to_snapshot<UIntMetric, uint64_t, true>(
        uint_metrics_,
        snapshot->uint_metrics,
        snapshot->uint_metric_timestamps,
        snapshot->metric_labels,
        &snapshot->uint_metric_units);
    process_metrics_to_snapshot<DoubleMetric, double, true>(
        double_metrics_,
        snapshot->double_metrics,
        snapshot->double_metric_timestamps,
        snapshot->metric_labels,
        &snapshot->double_metric_units);
    process_metrics_to_snapshot<StringMetric, std::string, true>(
        string_metrics_,
        snapshot->string_metrics,
        snapshot->string_metric_timestamps,
        snapshot->metric_labels,
        &snapshot->string_metric_units);

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
    int watchdog_timeout_seconds,
    int failure_exposure_duration_seconds,
    bool mmio_only) {
    try {
        Watchdog watchdog(watchdog_timeout_seconds);
        TT_FATAL(
            watchdog_timeout_seconds <= 0 || std::chrono::seconds(watchdog_timeout_seconds) > MONITOR_INTERVAL_SECONDS,
            "Watchdog timeout ({} seconds) cannot be shorter than the telemetry monitoring interval ({} seconds)",
            watchdog_timeout_seconds,
            MONITOR_INTERVAL_SECONDS.count());

        // Create TelemetryRunning system metric BEFORE UMD initialization
        bool_metrics_.push_back(std::make_unique<TelemetryRunningMetric>());

        // Keep index for updates (safe even if vector reallocates)
        size_t telemetry_running_index = bool_metrics_.size() - 1;

        std::unique_ptr<tt::umd::Cluster> cluster;
        std::unique_ptr<tt::tt_metal::Hal> hal;
        std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> psd;
        std::unique_ptr<TopologyHelper> topology_translation;

        // End time for main loop (std::nullopt means run forever)
        std::optional<std::chrono::steady_clock::time_point> loop_end_time;

        if (telemetry_enabled) {
            try {
                log_info(tt::LogAlways, "Initializing UMD and device metrics...");
                cluster = std::make_unique<tt::umd::Cluster>();
                auto distributed_context =
                    tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
                hal = create_hal(cluster);
                psd = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
                    cluster, distributed_context, hal.get(), rtoptions);
                topology_translation = std::make_unique<TopologyHelper>(cluster, psd, fsd);
                log_info(tt::LogAlways, "Created cluster, physical system descriptor, and HAL");
                log_info(tt::LogAlways, "Our hostname is: {}", topology_translation->my_host_name);

                auto arc_telemetry_reader_by_chip_id = create_arc_telemetry_readers(cluster);
                create_arc_metrics(
                    bool_metrics_,
                    uint_metrics_,
                    double_metrics_,
                    string_metrics_,
                    cluster,
                    topology_translation,
                    hal,
                    arc_telemetry_reader_by_chip_id);
                create_ethernet_metrics(
                    bool_metrics_,
                    uint_metrics_,
                    double_metrics_,
                    cluster,
                    fsd,
                    topology_translation,
                    hal,
                    arc_telemetry_reader_by_chip_id,
                    mmio_only);
                log_info(tt::LogAlways, "Initialized metrics");

                // Update TelemetryRunning metric to success state
                bool_metrics_[telemetry_running_index]->set_value(true);
            } catch (const std::exception& e) {
                log_fatal(tt::LogAlways, "UMD initialization failed: {}", e.what());

                // Mark telemetry as failed
                bool_metrics_[telemetry_running_index]->set_value(false);

                // Set end time for failure exposure period
                auto failure_duration = std::chrono::seconds(failure_exposure_duration_seconds);
                loop_end_time = std::chrono::steady_clock::now() + failure_duration;

                log_info(
                    tt::LogAlways,
                    "Exposing failure metric for {} seconds (until approximately {})",
                    failure_exposure_duration_seconds,
                    format_end_time(failure_duration));
            }
        } else {
            // Telemetry disabled - metric remains false (not running)
            log_info(tt::LogAlways, "Telemetry collection disabled");
        }

        // Create collection clients to connect to aggregate endpoints (if any specified)
        CollectionClients collection_clients(aggregate_endpoints, on_snapshot_received);
        log_info(tt::LogAlways, "Collection clients created successfully");

        // Get initial telemetry reading
        if (telemetry_enabled) {
            update(cluster);
            send_initial_snapshot(subscribers, topology_translation);
            log_info(tt::LogAlways, "Obtained initial readout and sent snapshot");
        }

        // Increment heartbeat after initialization
        watchdog.heartbeat();

        // Main telemetry monitoring loop
        // Continue until stopped or end time reached (if set)
        auto should_continue = [&]() {
            return !stopped_.load() &&
                   (!loop_end_time.has_value() || std::chrono::steady_clock::now() < loop_end_time.value());
        };
        while (should_continue()) {
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

        // If we exited due to end time (failure exposure complete), throw to exit process
        if (loop_end_time.has_value()) {
            log_fatal(tt::LogAlways, "Failure exposure period complete, exiting to allow orchestrator restart");
            throw std::runtime_error("UMD initialization failed - telemetry unavailable");
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
    int watchdog_timeout_seconds,
    int failure_exposure_duration_seconds,
    bool mmio_only) {
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
        watchdog_timeout_seconds,
        failure_exposure_duration_seconds,
        mmio_only);
    t.wait();
}
