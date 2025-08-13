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

#include <server/telemetry_provider.hpp>
#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>

static constexpr auto MONITOR_INTERVAL_SECONDS = std::chrono::seconds(5);

static std::mutex mtx_;
static std::queue<TelemetrySnapshot *> available_buffers_;

static std::atomic<bool> stopped_{false};

struct Metric {
    EthernetEndpoint ethernet_endpoint;
    bool is_up = false;
    bool dirty = false;

    // Construct full telemetry path identifier
    std::string telemetry_path() const {
        // Get hostname to prepend
        constexpr size_t hostname_buffer_size = 256; 
        std::vector<char> hostname_buffer(hostname_buffer_size);
        int hostname_size = gethostname(hostname_buffer.data(), hostname_buffer_size);

        // hostname + ethernet endpoint path
        std::vector<std::string> path_components{hostname_buffer.data()};
        auto ethernet_endpoint_path = ethernet_endpoint.telemetry_path();
        path_components.insert(path_components.end(), ethernet_endpoint_path.begin(), ethernet_endpoint_path.end());

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
};

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
            std::cout << "[TelemetryProvider] Released buffer" << std::endl;
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

static void sample(std::vector<Metric> &metrics, const tt::Cluster &cluster) {
    for (auto &metric: metrics) {
        bool is_up_now = is_ethernet_endpoint_up(cluster, metric.ethernet_endpoint);
        bool is_up_old = metric.is_up;
        metric.dirty = is_up_now != is_up_old;
        metric.is_up = is_up_now;
    }
}

static void send_initial_snapshot(const std::vector<std::shared_ptr<TelemetrySubscriber>> &subscribers, const std::vector<Metric> &metrics) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
    snapshot->is_absolute = true;
    
    for (size_t i = 0; i < metrics.size(); i++) {
        snapshot->metric_indices.push_back(i);
        snapshot->metric_names.push_back(metrics[i].telemetry_path());
        snapshot->metric_values.push_back(metrics[i].is_up);
    }

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void send_delta(const std::vector<std::shared_ptr<TelemetrySubscriber>> &subscribers, std::vector<Metric> &metrics) {
    std::shared_ptr<TelemetrySnapshot> snapshot = get_writeable_buffer();
    snapshot->is_absolute = false;

    for (size_t i = 0; i < metrics.size(); i++) {
        if (!metrics[i].dirty) {
            continue;
        }
        snapshot->metric_indices.push_back(i);
        snapshot->metric_values.push_back(metrics[i].is_up);
        metrics[i].dirty = false;
    }

    for (auto &subscriber: subscribers) {
        subscriber->on_telemetry_ready(snapshot);
    }
}

static void telemetry_thread(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers) {
    const tt::tt_metal::MetalContext &instance = tt::tt_metal::MetalContext::instance();
    const tt::Cluster &cluster = instance.get_cluster();

    // Create a single vector of metrics we will monitor
    std::vector<Metric> metrics;
    for (const auto &[chip_id, endpoints]: get_ethernet_endpoints_by_chip(cluster)) {
        for (const auto &endpoint: endpoints) {
            metrics.push_back(Metric{ .ethernet_endpoint = endpoint });
        }
    }

    // Continuously monitor on a loop
    sample(metrics, cluster);
    send_initial_snapshot(subscribers, metrics);
    while (!stopped_.load()) {
        std::this_thread::sleep_for(MONITOR_INTERVAL_SECONDS);
        sample(metrics, cluster);
        send_delta(subscribers, metrics);
    }
}

void run_telemetry_provider(std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers) {
    auto t = std::async(std::launch::async, telemetry_thread, subscribers);
    t.wait();
}