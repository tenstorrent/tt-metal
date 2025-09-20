// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <queue>
#include <unistd.h>

#include <tt-logger/tt-logger.hpp>

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
        [](TelemetrySnapshot *buffer) {
            // Custom deleter: do not delete, just return to pool. We use shared_ptr for its
            // thread-safe reference counting, allowing a buffer to be passed to multiple
            // consumers.
            log_debug(tt::LogAlways, "TelemetryProvider: Returned buffer");
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

static void update(
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics,
    const std::unique_ptr<tt::umd::Cluster>& cluster) {
    log_info(tt::LogAlways, "Starting telemetry readout...");
    std::chrono::steady_clock::time_point start_of_update_cycle = std::chrono::steady_clock::now();

    for (auto &metric: bool_metrics) {
        metric->update(cluster, start_of_update_cycle);
    }

    for (auto &metric: uint_metrics) {
        metric->update(cluster, start_of_update_cycle);
    }

    for (auto& metric : double_metrics) {
        metric->update(cluster, start_of_update_cycle);
    }

    std::chrono::steady_clock::time_point end_of_update_cycle = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_of_update_cycle - start_of_update_cycle).count();
    log_info(tt::LogAlways, "Telemetry readout took {} ms", duration_ms);
}

static void send_initial_snapshot(
    const std::vector<std::shared_ptr<TelemetrySubscriber>>& subscribers,
    const std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    const std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    const std::vector<std::unique_ptr<DoubleMetric>>& double_metrics) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();

    for (size_t i = 0; i < bool_metrics.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*bool_metrics[i]);
        size_t id = bool_metrics[i]->id;
        snapshot->bool_metric_ids.push_back(id);
        snapshot->bool_metric_names.push_back(path);
        snapshot->bool_metric_values.push_back(bool_metrics[i]->value());
        snapshot->bool_metric_timestamps.push_back(bool_metrics[i]->timestamp());
    }

    for (size_t i = 0; i < uint_metrics.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*uint_metrics[i]);
        size_t id = uint_metrics[i]->id;
        snapshot->uint_metric_ids.push_back(id);
        snapshot->uint_metric_names.push_back(path);
        snapshot->uint_metric_units.push_back(static_cast<uint16_t>(uint_metrics[i]->units));
        snapshot->uint_metric_values.push_back(uint_metrics[i]->value());
        snapshot->uint_metric_timestamps.push_back(uint_metrics[i]->timestamp());
    }

    for (size_t i = 0; i < double_metrics.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*double_metrics[i]);
        size_t id = double_metrics[i]->id;
        snapshot->double_metric_ids.push_back(id);
        snapshot->double_metric_names.push_back(path);
        snapshot->double_metric_units.push_back(static_cast<uint16_t>(double_metrics[i]->units));
        snapshot->double_metric_values.push_back(double_metrics[i]->value());
        snapshot->double_metric_timestamps.push_back(double_metrics[i]->timestamp());
    }

    // Populate unit label maps when names are populated
    snapshot->metric_unit_display_label_by_code = create_metric_unit_display_label_map();
    snapshot->metric_unit_full_label_by_code = create_metric_unit_full_label_map();

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void send_delta(
    const std::vector<std::shared_ptr<TelemetrySubscriber>>& subscribers,
    std::vector<std::unique_ptr<BoolMetric>>& bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>>& uint_metrics,
    std::vector<std::unique_ptr<DoubleMetric>>& double_metrics) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();

    for (size_t i = 0; i < bool_metrics.size(); i++) {
        if (!bool_metrics[i]->changed_since_transmission()) {
            continue;
        }
        snapshot->bool_metric_ids.push_back(bool_metrics[i]->id);
        snapshot->bool_metric_values.push_back(bool_metrics[i]->value());
        snapshot->bool_metric_timestamps.push_back(bool_metrics[i]->timestamp());
        bool_metrics[i]->mark_transmitted();
    }

    for (size_t i = 0; i < uint_metrics.size(); i++) {
        if (!uint_metrics[i]->changed_since_transmission()) {
            continue;
        }
        snapshot->uint_metric_ids.push_back(uint_metrics[i]->id);
        snapshot->uint_metric_values.push_back(uint_metrics[i]->value());
        snapshot->uint_metric_timestamps.push_back(uint_metrics[i]->timestamp());
        uint_metrics[i]->mark_transmitted();
    }

    for (size_t i = 0; i < double_metrics.size(); i++) {
        if (!double_metrics[i]->changed_since_transmission()) {
            continue;
        }
        snapshot->double_metric_ids.push_back(double_metrics[i]->id);
        snapshot->double_metric_values.push_back(double_metrics[i]->value());
        snapshot->double_metric_timestamps.push_back(double_metrics[i]->timestamp());
        double_metrics[i]->mark_transmitted();
    }

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void telemetry_thread(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers) {
    std::unique_ptr<tt::umd::Cluster> cluster = std::make_unique<tt::umd::Cluster>();
    std::unique_ptr<tt::tt_metal::Hal> hal = create_hal(cluster);
    log_info(tt::LogAlways, "Created cluster and HAL");

    // Create vectors of all metrics we will monitor by value type
    size_t id = 1;
    std::vector<std::unique_ptr<BoolMetric>> bool_metrics;
    std::vector<std::unique_ptr<UIntMetric>> uint_metrics;
    std::vector<std::unique_ptr<DoubleMetric>> double_metrics;

    // Create Ethernet metrics
    for (const auto &[chip_id, endpoints]: get_ethernet_endpoints_by_chip(cluster)) {
        for (const auto &endpoint: endpoints) {
            bool_metrics.push_back(std::make_unique<EthernetEndpointUpMetric>(id++, endpoint, hal));
            uint_metrics.push_back(std::make_unique<EthernetRetrainCountMetric>(id++, endpoint, cluster, hal));
            if (hal->get_arch() == tt::ARCH::WORMHOLE_B0) {
                // These are available only on Wormhole
                uint_metrics.push_back(std::make_unique<EthernetCRCErrorCountMetric>(id++, endpoint, cluster, hal));
                uint_metrics.push_back(std::make_unique<EthernetCorrectedCodewordCountMetric>(id++, endpoint, cluster, hal));
                uint_metrics.push_back(std::make_unique<EthernetUncorrectedCodewordCountMetric>(id++, endpoint, cluster, hal));
            }
        }
    }
    log_info(tt::LogAlways, "Created Ethernet metrics");

    // Create ARC telemetry metrics for MMIO-capable chips
    for (const auto& [chip_identifier, reader] : create_arc_telemetry_readers_for_mmio_chips(cluster)) {
        bool_metrics.push_back(std::make_unique<ARCTelemetryAvailableMetric>(id++, reader));

        // Create UInt metrics with appropriate masks and units
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::AICLK, "AIClock", 0xffff, MetricUnit::MEGAHERTZ));
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::AXICLK, "AXIClock", 0xffffffff, MetricUnit::MEGAHERTZ));
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::ARCCLK, "ARCClock", 0xffffffff, MetricUnit::MEGAHERTZ));
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::FAN_SPEED, "FanSpeed", 0xffffffff, MetricUnit::REVOLUTIONS_PER_MINUTE));
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::TDP, "TDP", 0xffff, MetricUnit::WATTS));
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::TDC, "TDC", 0xffff, MetricUnit::AMPERES));
        uint_metrics.push_back(std::make_unique<ARCUintMetric>(id++, reader, tt::umd::TelemetryTag::VCORE, "VCore", 0xffffffff, MetricUnit::MILLIVOLTS));

        // Create Double metrics with appropriate masks, scale factors, and units
        // For ASIC temperature, check architecture to determine mask and scale factor
        uint32_t asic_temp_mask;
        double asic_temp_scale;
        if (reader->get_arch() == tt::ARCH::BLACKHOLE) {
            asic_temp_mask = 0xffffffff;
            asic_temp_scale = 1.0/65536.0;
        } else {
            asic_temp_mask = 0xffff;
            asic_temp_scale = 1.0/16.0;
        }
        double_metrics.push_back(std::make_unique<ARCDoubleMetric>(id++, reader, tt::umd::TelemetryTag::ASIC_TEMPERATURE, "ASICTemperature", asic_temp_mask, asic_temp_scale, MetricUnit::CELSIUS, ARCDoubleMetric::Signedness::SIGNED));
        double_metrics.push_back(std::make_unique<ARCDoubleMetric>(id++, reader, tt::umd::TelemetryTag::BOARD_TEMPERATURE, "BoardTemperature", 0xffffffff, 1.0/65536.0, MetricUnit::CELSIUS, ARCDoubleMetric::Signedness::SIGNED));
    }
    log_info(tt::LogAlways, "Created ARC metrics");
    log_info(tt::LogAlways, "Initialized telemetry thread");

    // Continuously monitor on a loop
    update(bool_metrics, uint_metrics, double_metrics, cluster);
    send_initial_snapshot(subscribers, bool_metrics, uint_metrics, double_metrics);
    log_info(tt::LogAlways, "Obtained initial readout and sent snapshot");
    while (!stopped_.load()) {
        std::this_thread::sleep_for(MONITOR_INTERVAL_SECONDS);
        update(bool_metrics, uint_metrics, double_metrics, cluster);
        send_delta(subscribers, bool_metrics, uint_metrics, double_metrics);
    }

    log_info(tt::LogAlways, "Telemetry thread stopped");
}

void run_telemetry_provider(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers) {
    // Prefill hostname
    gethostname(hostname_, sizeof(hostname_));

    // Run telemetry thread
    auto t = std::async(std::launch::async, telemetry_thread, subscribers);
    t.wait();
}
