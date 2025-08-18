/*
 * TODO:
 * -----
 * - Wait until subscribers have all finished before fetching a new buffer. Continue to use
 *   existing buffer until ready to hand off.
 * - Get rid of is_absolute. Contract should simply be that if names have been transmitted,
 *   all data represents new metrics. Otherwise, the snapshot contains delta metrics (of existing
 *   metrics).
 * - Hostname may contain "_" character, so using this as a path component delimiter is unsafe.
 *   We should instead use either a different separator (like "/") or encode paths as vectors.
 *   For future exporters that need to export a single string, these can be formed when exporting.
 */

#include <future>
#include <queue>
#include <unistd.h>

#include "impl/context/metal_context.hpp"

#include <telemetry/telemetry_provider.hpp>
#include <telemetry/ethernet/ethernet_metrics.hpp>

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
            std::cout << "[TelemetryProvider] Returned buffer" << std::endl;
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
        std::cout << "[TelemetryProvider] Got buffer from pool" << std::endl;
    } else {
        // Pool exhausted, create new buffer
        buffer = new TelemetrySnapshot();
        std::cout << "[TelemetryProvider] Allocated new buffer" << std::endl;
    }

    // Ensure it is clear
    buffer->clear();

    // Return a RAII handle that will automatically return buffer to pool
    return create_new_handoff_buffer(buffer);
}

static void update(
    std::vector<std::unique_ptr<BoolMetric>> &bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>> &uint_metrics,
    const tt::Cluster &cluster
) {
    for (auto &metric: bool_metrics) {
        metric->update(cluster);
    }

    for (auto &metric: uint_metrics) {
        metric->update(cluster);
    }
}

static std::string get_cluster_wide_telemetry_path(const Metric &metric) {
    // Cluster-wide path is: hostname + metric path
    std::vector<std::string> path_components{static_cast<const char *>(hostname_)};
    auto local_path = metric.telemetry_path();
    path_components.insert(path_components.end(), local_path.begin(), local_path.end());

    // Join with '_'
    std::string path;
    for (auto it = path_components.begin(); it != path_components.end(); ) {
        path += *it;
        ++it;
        if (it != path_components.end()) {
            path += '_';
        }
    }
    return path;
}

static void send_initial_snapshot(
    const std::vector<std::shared_ptr<TelemetrySubscriber>> &subscribers,
    const std::vector<std::unique_ptr<BoolMetric>> &bool_metrics,
    const std::vector<std::unique_ptr<UIntMetric>> &uint_metrics
) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
    snapshot->is_absolute = true;

    for (size_t i = 0; i < bool_metrics.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*bool_metrics[i]);
        size_t id = bool_metrics[i]->id;
        snapshot->bool_metric_ids.push_back(id);
        snapshot->bool_metric_names.push_back(path);
        snapshot->bool_metric_values.push_back(bool_metrics[i]->value());
    }

    for (size_t i = 0; i < uint_metrics.size(); i++) {
        std::string path = get_cluster_wide_telemetry_path(*uint_metrics[i]);
        size_t id = uint_metrics[i]->id;
        snapshot->uint_metric_ids.push_back(id);
        snapshot->uint_metric_names.push_back(path);
        snapshot->uint_metric_values.push_back(uint_metrics[i]->value());
    }

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void send_delta(
    const std::vector<std::shared_ptr<TelemetrySubscriber>> &subscribers,
    std::vector<std::unique_ptr<BoolMetric>> &bool_metrics,
    std::vector<std::unique_ptr<UIntMetric>> &uint_metrics
) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
    snapshot->is_absolute = false;

    for (size_t i = 0; i < bool_metrics.size(); i++) {
        if (!bool_metrics[i]->changed_since_transmission()) {
            continue;
        }
        snapshot->bool_metric_ids.push_back(bool_metrics[i]->id);
        snapshot->bool_metric_values.push_back(bool_metrics[i]->value());
        bool_metrics[i]->mark_transmitted();
    }

    for (size_t i = 0; i < uint_metrics.size(); i++) {
        if (!uint_metrics[i]->changed_since_transmission()) {
            continue;
        }
        snapshot->uint_metric_ids.push_back(uint_metrics[i]->id);
        snapshot->uint_metric_values.push_back(uint_metrics[i]->value());
        uint_metrics[i]->mark_transmitted();
    }

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void telemetry_thread(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers) {
    const tt::tt_metal::MetalContext &instance = tt::tt_metal::MetalContext::instance();
    const tt::Cluster &cluster = instance.get_cluster();

    // Create vectors of all metrics we will monitor by value type
    size_t id = 1;
    std::vector<std::unique_ptr<BoolMetric>> bool_metrics;
    std::vector<std::unique_ptr<UIntMetric>> uint_metrics;
    for (const auto &[chip_id, endpoints]: get_ethernet_endpoints_by_chip(cluster)) {
        for (const auto &endpoint: endpoints) {
            bool_metrics.push_back(std::make_unique<EthernetEndpointUpMetric>(id++, endpoint));
            uint_metrics.push_back(std::make_unique<EthernetCRCErrorCountMetric>(id++, endpoint, cluster));
            uint_metrics.push_back(std::make_unique<EthernetRetrainCountMetric>(id++, endpoint, cluster));
            uint_metrics.push_back(std::make_unique<EthernetCorrectedCodewordCountMetric>(id++, endpoint, cluster));
            uint_metrics.push_back(std::make_unique<EthernetUncorrectedCodewordCountMetric>(id++, endpoint, cluster));
        }
    }

    // Continuously monitor on a loop
    update(bool_metrics, uint_metrics, cluster);
    send_initial_snapshot(subscribers, bool_metrics, uint_metrics);
    while (!stopped_.load()) {
        std::this_thread::sleep_for(MONITOR_INTERVAL_SECONDS);
        update(bool_metrics, uint_metrics,cluster);
        send_delta(subscribers, bool_metrics, uint_metrics);
    }
}

void run_telemetry_provider(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers) {
    // Prefill hostname
    gethostname(hostname_, sizeof(hostname_));

    // Run telemetry thread
    auto t = std::async(std::launch::async, telemetry_thread, subscribers);
    t.wait();
}
