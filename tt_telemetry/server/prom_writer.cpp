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

class PromWriter : public TelemetrySubscriber {
private:
    std::filesystem::path prom_file_path_;
    std::mutex file_mutex_;
    std::thread processing_thread_;
    std::atomic<bool> running_{false};
    std::queue<std::shared_ptr<TelemetrySnapshot>> pending_snapshots_;
    std::mutex snapshot_mutex_;
    std::condition_variable snapshot_cv_;
    std::unordered_set<std::string> documented_metrics_;
    static constexpr std::chrono::milliseconds snapshot_check_timeout_{1000};

    void add_metric_comments_once(std::ostream& out, std::string_view metric_name, std::string_view help_text) {
        if (documented_metrics_.insert(std::string(metric_name)).second) {
            // First time this metric is seen
            out << "# HELP " << metric_name << " " << help_text << "\n"; // TODO(kkfernandez): for now these are highly uninformative
            out << "# TYPE " << metric_name << " gauge\n";
        }
    }

    std::string sanitize_metric_name(std::string_view path) {
        // Prefix with a standard namespace
        std::string sanitized_prefix = "tt_metal_";
        
        // Convert to valid Prometheus metric name
        std::string sanitized_name;
        
        // Ensure first character is a letter or underscore
        if (path.empty()) {
            return sanitized_prefix + "unnamed_metric";
        }
        
        // First character must be a letter or underscore
        if (!std::isalpha(path[0]) && path[0] != '_') {
            sanitized_name = '_';
        }
        
        // Subsequent characters can be alphanumeric or underscore
        for (char c : path) {
            if (std::isalnum(c) || c == '_' || c == ':') {
                sanitized_name += c;
            } else {
                // Replace unsupported characters with underscore
                sanitized_name += '_';
            }
        }
        
        return sanitized_prefix + sanitized_name;
    }

    std::string format_metric_line(
        std::string_view name, 
        std::string_view value,
        const std::unordered_map<std::string, uint64_t>& timestamps
    ) {
        std::string sanitized_name = sanitize_metric_name(name);
        
        // Check if we have a timestamp for this metric
        auto timestamp_it = timestamps.find(std::string(name));
        std::string timestamp_suffix = timestamp_it != timestamps.end() 
            ? " " + std::to_string(timestamp_it->second) 
            : "";
        
        return sanitized_name + " " + std::string(value) + timestamp_suffix + "\n";
    }

    std::string convert_to_prom_format(const TelemetrySnapshot& snapshot) {
        std::stringstream prom_output;

        for (const auto& [path, value] : snapshot.bool_metrics) {
            write_prom_metric(
                prom_output,
                path,
                std::to_string(value ? 1 : 0),
                snapshot.bool_metric_timestamps,
                "Boolean metric from Tenstorrent Metal",
                ""  // No unit label
            );
        }

        process_prom_metrics(snapshot.uint_metrics,
                            snapshot.uint_metric_units,
                            snapshot.uint_metric_timestamps,
                            snapshot,
                            prom_output,
                            "Unsigned integer metric from Tenstorrent Metal");

        process_prom_metrics(snapshot.double_metrics,
                            snapshot.double_metric_units,
                            snapshot.double_metric_timestamps,
                            snapshot,
                            prom_output,
                            "Floating-point metric from Tenstorrent Metal");

        return prom_output.str();
    }

    void write_prom_metric(std::ostream& out,
                        std::string_view path,
                        std::string_view value_str,
                        const std::unordered_map<std::string, uint64_t>& timestamps,
                        std::string_view help_text,
                        std::string_view label) {
        std::string sanitized_name = sanitize_metric_name(path);
        add_metric_comments_once(out, sanitized_name, help_text);

        out << sanitized_name;
        if (!label.empty()) {
            out << "{" << label << "}";
        }

        out << " " << value_str;

        auto it = timestamps.find(std::string(path));
        if (it != timestamps.end()) {
            out << " " << it->second;
        }

        out << "\n";
    }

    template <typename T>
    void process_prom_metrics(const std::unordered_map<std::string, T>& metrics,
                            const std::unordered_map<std::string, uint16_t>& metric_units,
                            const std::unordered_map<std::string, uint64_t>& timestamps,
                            const TelemetrySnapshot& snapshot,
                            std::ostream& out,
                            std::string_view help_text) {
        for (const auto& [path, value] : metrics) {
            std::string label;

            auto unit_it = metric_units.find(path);
            if (unit_it != metric_units.end()) {
                auto label_it = snapshot.metric_unit_display_label_by_code.find(unit_it->second);
                if (label_it != snapshot.metric_unit_display_label_by_code.end()) {
                    label = "unit=\"" + label_it->second + "\"";
                }
            }

            write_prom_metric(out, path, std::to_string(value), timestamps, help_text, label);
        }
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

    // Adds type info of the metrics to the file's header
    bool ensure_metric_comments_exist() {
        std::ifstream check_file(prom_file_path_);
        std::string first_line;
        std::getline(check_file, first_line);
        
        // If file is empty or doesn't have comments, add them
        if (first_line.empty() || !first_line.starts_with("# HELP")) {
            std::ofstream comments_file(prom_file_path_, std::ios::app);
            comments_file << "# HELP tt_metal_bool_metrics Boolean metrics from Tenstorrent Metal\n";
            comments_file << "# TYPE tt_metal_bool_metrics gauge\n";
            
            comments_file << "# HELP tt_metal_uint_metrics Unsigned integer metrics from Tenstorrent Metal\n";
            comments_file << "# TYPE tt_metal_uint_metrics gauge\n";
            
            comments_file << "# HELP tt_metal_double_metrics Floating-point metrics from Tenstorrent Metal\n";
            comments_file << "# TYPE tt_metal_double_metrics gauge\n";
            
            return true;
        }
        return false;
    }

    // Writes metrics to file
    bool write_metrics_to_file(std::string_view metrics) {
        ensure_metric_comments_exist();
        
        std::lock_guard<std::mutex> lock(file_mutex_);

        if (!std::filesystem::exists(prom_file_path_)) {
            log_warning(tt::LogAlways, "Prometheus metrics file not found. Recreating: {}", prom_file_path_.string());
            initialize_file();
        }
    
        // Open file in append mode
        std::ofstream prom_file(prom_file_path_, std::ios::app | std::ios::out);

        if (!prom_file.is_open()) {
            log_error(tt::LogAlways, "Failed to open Prometheus metrics file: {}", prom_file_path_.string());
            return false;
        }

        try {
            // Write metrics
            prom_file << metrics;
            prom_file.flush();
            
            if (prom_file.fail()) {
                log_error(tt::LogAlways, "Write failed for Prometheus metrics file: {}", prom_file_path_.string());
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            log_error(tt::LogAlways, "Exception while writing to Prometheus metrics file: {}", e.what());
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

            auto snapshot = get_next_snapshot(lock);
            lock.unlock();

            // Skip if no snapshot (spurious wakeup or woken on timeout)
            if (!snapshot) {
                continue;
            }
            
            try {
                // Convert to Prometheus format
                auto prom_metrics = convert_to_prom_format(*snapshot);
                
                // Write to file
                if (!write_metrics_to_file(prom_metrics)) {
                    // TODO(kkfernandez): Consider additional error handling if file write fails
                }
            } catch (const std::exception& e) {
                log_error(tt::LogAlways, "Error processing telemetry for Prometheus: {}", e.what());
            }
        }
    }

    void initialize_file() {
        std::ofstream initial_file(prom_file_path_, std::ios::out);
        if (!initial_file.is_open()) {
            throw std::runtime_error("Unable to create Prometheus metrics file: " + prom_file_path_.string());
        }

        // Obtain human-readable timestamp
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        std::string time_str = std::ctime(&now_c);
        time_str.pop_back(); // Remove trailing newline

        // Write header
        initial_file << "# Tenstorrent Metal Telemetry Metrics\n";
        initial_file << "# Start Time: " << time_str << "\n\n";
    }

public:
    PromWriter(std::string_view file_path) : prom_file_path_(file_path) {
        // Ensure directory exists
        std::filesystem::create_directories(prom_file_path_.parent_path());

        initialize_file();
    }

    void start() {
        running_ = true;
        processing_thread_ = std::thread(&PromWriter::process_telemetry, this);
    }

    void stop() {
        running_ = false;
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