// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <fstream>
#include <mutex>
#include <thread>
#include <queue>
#include <chrono>
#include <memory>
#include <filesystem>
#include <string_view>
#include <condition_variable>
#include <telemetry/telemetry_subscriber.hpp>
#include <server/prom_writer.hpp>
#include <unistd.h>

class PromWriter : public TelemetrySubscriber {
private:
    std::filesystem::path prom_file_path_;
    std::mutex file_mutex_;
    std::thread processing_thread_;
    std::atomic<bool> running_{false};
    std::queue<std::shared_ptr<TelemetrySnapshot>> pending_snapshots_;
    std::mutex snapshot_mutex_;
    std::condition_variable snapshot_cv_;
    std::string system_info_;
    const std::string hostname_;

    // Current state of all metrics
    struct MetricState {
        struct MetricData {
            std::string value;
            uint64_t timestamp;
            std::string unit_label;
            std::string help_text;
        };
        std::unordered_map<std::string, MetricData> metrics;
    };
    MetricState current_state_;
    std::mutex state_mutex_;

    static constexpr std::chrono::milliseconds snapshot_check_timeout_{1000};

    static std::string get_hostname() {
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) != 0) {
            throw std::runtime_error("Failed to get hostname");
        }
        std::string hostname_str(hostname);
        return hostname_str;
    }

    std::string sanitize_metric_name(std::string_view path) {
        std::string path_str(path);

        // Find and remove hostname from the path
        size_t hostname_pos = path_str.find(hostname_);
        if (hostname_pos == std::string::npos) {
            throw std::runtime_error("Metric path does not contain hostname '" + hostname_ + "': " + path_str);
        }

        // Skip hostname and the expected slash after it
        size_t after_hostname = hostname_pos + hostname_.length();
        if (after_hostname >= path_str.length() || path_str[after_hostname] != '/') {
            throw std::runtime_error("Expected slash after hostname in metric path: " + path_str);
        }
        after_hostname++;  // Skip the slash
        path_str = path_str.substr(after_hostname);

        // Convert to valid Prometheus metric name
        std::string sanitized;  // TODO(kkfernandez): this could be prefixed with "tt_metal_"
        for (char c : path_str) {
            if (std::isalnum(c) || c == '_') {
                sanitized += c;
            } else {
                sanitized += '_';
            }
        }

        return sanitized;
    }

    void update_state_from_snapshot(const TelemetrySnapshot& snapshot) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Helper lambda to process all metric types
        auto process_metrics = [&](const auto& metrics,
                                   const auto& timestamps,
                                   const auto* units,  // nullable for bool metrics
                                   const std::string& help_text,
                                   auto value_converter) {
            for (const auto& [path, value] : metrics) {
                MetricState::MetricData data;
                data.value = value_converter(value);
                data.help_text = help_text;

                // Get timestamp
                auto ts_it = timestamps.find(path);
                data.timestamp = (ts_it != timestamps.end()) ? ts_it->second : 0;

                // Get unit label if units map is provided
                if constexpr (!std::is_null_pointer_v<decltype(units)>) {
                    if (units) {
                        auto unit_it = units->find(path);
                        if (unit_it != units->end()) {
                            auto label_it = snapshot.metric_unit_display_label_by_code.find(unit_it->second);
                            // Throw an error if no label is found
                            if (label_it == snapshot.metric_unit_display_label_by_code.end()) {
                                throw std::runtime_error(
                                    "No display label found for unit code: " + std::to_string(unit_it->second) +
                                    " for metric path: " + path);
                            }
                            data.unit_label = label_it->second;
                        }
                    }
                }

                current_state_.metrics[sanitize_metric_name(path)] = data;
            }
        };

        // Process bool metrics
        process_metrics(
            snapshot.bool_metrics,
            snapshot.bool_metric_timestamps,
            static_cast<const std::map<std::string, int>*>(nullptr),  // no units for bool metrics
            "Boolean metric from Tenstorrent Metal",  // TODO(kkfernandez): something more descriptive, per metric
            [](bool v) { return std::to_string(v ? 1 : 0); });

        // Process unsigned integer metrics
        process_metrics(
            snapshot.uint_metrics,
            snapshot.uint_metric_timestamps,
            &snapshot.uint_metric_units,
            "Unsigned integer metric from Tenstorrent Metal",  // TODO(kkfernandez): something more descriptive, per
                                                               // metric
            [](auto v) { return std::to_string(v); });

        // Process floating-point metrics
        process_metrics(
            snapshot.double_metrics,
            snapshot.double_metric_timestamps,
            &snapshot.double_metric_units,
            "Floating-point metric from Tenstorrent Metal",  // TODO(kkfernandez): something more descriptive, per
                                                             // metric
            [](auto v) { return std::to_string(v); });
    }

    std::string generate_prom_output() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        std::stringstream output;

        // Write header
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_c);
        time_str.pop_back(); // Remove trailing newline

        output << "# Tenstorrent Metal Telemetry Metrics\n";
        output << "# Host: " << hostname_ << "\n";
        output << "# Generated at: " << time_str << "\n\n";

        // Write all metrics with their metadata
        for (const auto& [metric_name, data] : current_state_.metrics) {
            // Write HELP and TYPE
            output << "# HELP " << metric_name << " " << data.help_text << "\n";
            output << "# TYPE " << metric_name << " gauge\n";

            // Write metric line
            output << metric_name;
            if (!data.unit_label.empty()) {
                output << "{unit=\"" << data.unit_label << "\"}";
            }
            output << " " << data.value;
            if (data.timestamp > 0) {
                output << " " << data.timestamp;
            }
            output << "\n\n";
        }

        return output.str();
    }

    // Get snapshot if one is ready
    std::shared_ptr<TelemetrySnapshot> get_next_snapshot(std::unique_lock<std::mutex>& lock) {
        if (pending_snapshots_.empty()) {
            return nullptr;
        }

        auto snapshot = std::move(pending_snapshots_.front());
        pending_snapshots_.pop();
        return snapshot;
    }

    // Writes metrics to file (complete overwrite)
    bool try_write_metrics_to_file() {
        std::string content = generate_prom_output();

        std::lock_guard<std::mutex> lock(file_mutex_);

        // Write to temp file first, then rename (atomic operation)
        std::filesystem::path temp_path = prom_file_path_.string() + ".tmp";
        std::ofstream prom_file(temp_path, std::ios::out | std::ios::trunc);

        if (!prom_file.is_open()) {
            log_error(tt::LogAlways, "Failed to open temporary Prometheus metrics file: {}", temp_path.string());
            return false;
        }

        try {
            prom_file << content;
            prom_file.close();

            if (prom_file.fail()) {
                log_error(tt::LogAlways, "Write failed for Prometheus metrics file: {}", temp_path.string());
                std::filesystem::remove(temp_path);
                return false;
            }

            // Atomically rename temp file to actual file
            try {
                std::filesystem::rename(temp_path, prom_file_path_);
            } catch (const std::filesystem::filesystem_error& e) {
                // If rename fails (e.g., across filesystems), fall back to copy+remove
                std::filesystem::copy_file(
                    temp_path, prom_file_path_, std::filesystem::copy_options::overwrite_existing);
                std::filesystem::remove(temp_path);
            }
            return true;
        } catch (const std::exception& e) {
            log_error(tt::LogAlways, "Exception while writing to Prometheus metrics file: {}", e.what());
            try {
                std::filesystem::remove(temp_path);
            } catch (...) {}
            return false;
        }
    }

    void process_telemetry() {
        while (running_) {
            // Wait until:
            // 1. A snapshot is available
            // 2. We're no longer running
            // 3. Timeout occurs
            // 4. Spurious wakeup
            std::unique_lock<std::mutex> lock(snapshot_mutex_);
            snapshot_cv_.wait_for(lock, snapshot_check_timeout_, [this]() {
                return !pending_snapshots_.empty() || !running_;
            });

            // Exit if not running
            if (!running_) {
                break;
            }

            // Process all pending snapshots
            while (!pending_snapshots_.empty()) {
                auto snapshot = get_next_snapshot(lock);
                lock.unlock();

                if (snapshot) {
                    try {
                        // Update internal state
                        update_state_from_snapshot(*snapshot);

                        // Write current complete state to file
                        try_write_metrics_to_file();
                    } catch (const std::exception& e) {
                        log_error(tt::LogAlways, "Error processing telemetry for Prometheus: {}", e.what());
                    }
                }

                lock.lock();
            }
        }
    }

public:
    PromWriter(std::string_view file_path) : prom_file_path_(file_path), hostname_(get_hostname()) {
        // Ensure directory exists
        std::filesystem::create_directories(prom_file_path_.parent_path());

        // Create an initial empty file
        if (!try_write_metrics_to_file()) {
            throw std::runtime_error("Failed to initialize Prometheus metrics file at: " + prom_file_path_.string());
        }
    }

    void start() {
        running_ = true;
        processing_thread_ = std::thread(&PromWriter::process_telemetry, this);
    }

    void stop() {
        running_ = false;
        snapshot_cv_.notify_one();
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

    void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) override {
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex_);
            pending_snapshots_.emplace(std::move(telemetry));
        }
        snapshot_cv_.notify_one();
    }

    ~PromWriter() {
        stop();
    }
};

// Similar to other server creation functions
std::pair<std::future<bool>, std::shared_ptr<TelemetrySubscriber>> run_prom_writer(
    std::string_view file_path) {
    auto writer = std::make_shared<PromWriter>(file_path);
    auto subscriber = std::static_pointer_cast<TelemetrySubscriber>(writer);

    auto future = std::async(std::launch::async, [writer]() {
        try {
            writer->start();
            return true;
        } catch (const std::exception& e) {
            log_fatal(tt::LogAlways, "Prometheus writer error: {}", e.what());
            return false;
        }
    });
    return std::make_pair(std::move(future), subscriber);
}
